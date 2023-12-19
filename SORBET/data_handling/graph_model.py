from dataclasses import dataclass
import pickle
import numpy as np
import networkx as nx

class OmicsGraph:

    def __init__(self, vertex_lst: list, edge_lst: list, data: np.ndarray, markers: list, graph_label: int, meta_markers: list = None):
        self.graph = nx.Graph()
        self.edge_lst = edge_lst
        self.node_attributes = dict()
        for vi, vertex in enumerate(vertex_lst):
            self.graph.add_node(vertex) 
            
            marker_mapping = {marker:value for marker, value in zip(markers, data[vi])}
            self.node_attributes[vertex] = marker_mapping

        self.graph.add_edges_from(edge_lst)
        
        self.markers = markers
        self.vertices = vertex_lst
        self.graph_label = graph_label

        if meta_markers is not None:
            self.meta_markers = meta_markers[0]
            self.meta_marker_data = meta_markers[1]

            self.node_meta_attributes = dict()
            for vi, vertex in enumerate(vertex_lst):
                meta_marker_mapping = {marker:value for marker, value in zip(self.meta_markers, self.meta_marker_data[vi])} 
                self.node_meta_attributes[vertex] = meta_marker_mapping
        else:
            self.meta_markers = None
            self.meta_marker_data = None
            self.node_meta_attributes = None

    def get_marker(self, marker: str, nodes: list = None) -> np.ndarray:
        if marker in self.markers:
            attrs = self.node_attributes
        else:
            assert self.meta_markers is not None and marker in self.meta_markers
            attrs = self.node_meta_attributes

        if nodes is not None:
            marker_vals = [attrs[vi][marker] for vi in nodes]
            return nodes, marker_vals 
        else:
            marker_vals = [attrs[vi][marker] for vi in self.vertices]
            return self.vertices, marker_vals
    
    def get_khop_neighborhood(self, vertex, k: int) -> np.ndarray:
        neighbors = set(self.graph.neighbors(vertex))
        
        new_neighbors = set(neighbors)
        for kh in range(k - 1):
            if len(new_neighbors) == 0: break

            next_hop = set.union(*(set(self.graph.neighbors(vi)) for vi in new_neighbors))
            new_neighbors = next_hop.difference(neighbors)
            neighbors |= next_hop 

        return list(neighbors)
    
    def make_subgraph(self, vertex_lst: list):
        subgraph = self.graph.subgraph(vertex_lst)
        
        V = list(subgraph.nodes())
        E = list(subgraph.edges())
        
        X = list() 
        for vi in V:
            X.append([self.node_attributes[vi][marker] for marker in self.markers])
        X = np.array(X)
        
        if self.meta_markers is not None:
            M = list()
            for vi in V:
                M.append([self.node_meta_attributes[vi][marker] for marker in self.meta_markers])
            meta_marker_data = (self.meta_markers, np.array(M))
        else:
            meta_marker_data = None

        return OmicsGraph(V, E, X, self.markers, self.graph_label, meta_marker_data)

    def get_node_data(self):
        nodes_data = list()
        for vertex in self.vertices:
            data_arr = [self.node_attributes[vertex][marker] for marker in self.markers]
            nodes_data.append(data_arr)
        
        return np.array(nodes_data)

    def set_node_data(self, node_data: np.ndarray, marker_update: list = None):
        assert len(self.vertices) == node_data.shape[0]
        if marker_update is not None:
            assert len(marker_update) == node_data.shape[1]
            self.markers = marker_update
        else:
            assert len(self.markers) == node_data.shape[1]

        for vertex, arr in zip(self.vertices, node_data):
            marker_mapping = {marker:val for marker, val in zip(self.markers, arr)}
            self.node_attributes[vertex] = marker_mapping 

def load_omicsgraph(fpath: str) -> OmicsGraph:
    with open(fpath, 'rb') as ifile:
        idata = pickle.load(ifile)
        
        vertex_lst = idata[0][1]
        edge_lst = idata[1][1]
        marker_lst = idata[2][1]
        marker_data = idata[3][1]
        graph_label = idata[4][1]
        
        meta_marker_data = idata[5][1] 

    return OmicsGraph(vertex_lst, edge_lst, marker_data, marker_lst, graph_label, meta_marker_data)

def dump_omicsgraph(input_graph: OmicsGraph, fpath: str) -> None:
    vertex_lst = input_graph.vertices
    marker_lst = input_graph.markers
    edge_lst = list(input_graph.graph.edges())

    data = [[input_graph.node_attributes[vi][mj] for mj in marker_lst] for vi in vertex_lst]
    
    graph_label = input_graph.graph_label
    
    if input_graph.meta_markers is not None:
        meta_marker_data = [input_graph.meta_markers, input_graph.meta_marker_data]
    else:
        meta_marker_data = None

    with open(fpath, 'wb+') as ofile:
        odata = [
            ["vertices", vertex_lst],
            ["edges", edge_lst],
            ["markers", marker_lst],
            ["marker_data", data],
            ["graph_label", graph_label],
            ["meta_markers", meta_marker_data] 
        ]

        pickle.dump(odata, ofile)
