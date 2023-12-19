import os, sys
import csv

import numpy as np
import torch
from torch_geometric.data import Data

from .graph_model import OmicsGraph, load_omicsgraph, dump_omicsgraph
from .subgraph_extraction import subgraph_extraction

def create_subgraphs(complete_graphs_dirpath: str, output_dirpath: str, subgraph_extraction_algorithm: str, subgraph_extraction_algorithm_kwargs: dict, 
        mapping_fname: str = "py_subgraph_mapping.csv", subgraph_dirname: str = "graphs_py", graphs_for_node_data_dirpath: str = None):
    
    subgraph_file_mapping = list()
    
    pygraphs_output_dirpath = os.path.join(output_dirpath, subgraph_dirname)
    if not os.path.exists(pygraphs_output_dirpath):
        os.makedirs(pygraphs_output_dirpath)

    for ifile in os.listdir(complete_graphs_dirpath):
        fpath = os.path.join(complete_graphs_dirpath, ifile)
        graph = load_omicsgraph(fpath)
        graph_id = os.path.splitext(ifile)[0]
        if graphs_for_node_data_dirpath is not None:
            fpath_for_node_data = os.path.join(graphs_for_node_data_dirpath, ifile)
            graph_for_node_data = load_omicsgraph(fpath_for_node_data)
        else:
            graph_for_node_data = None
        subgraph_files = list()
        
        subgraphs = subgraph_extraction(graph, subgraph_extraction_algorithm, subgraph_extraction_algorithm_kwargs, graph_for_node_data)
        for sg_idx, sg in enumerate(subgraphs):
            fname = f'{graph_id}_sg_{sg_idx}.p'
            ofpath = os.path.join(pygraphs_output_dirpath, fname)
            dump_omicsgraph(sg, ofpath)
    
            subgraph_files.append(os.path.join(subgraph_dirname, fname))
        if graphs_for_node_data_dirpath is not None:
            subgraph_file_mapping.append([graph_id, fpath_for_node_data, *subgraph_files])
        else:
            subgraph_file_mapping.append([graph_id, fpath, *subgraph_files])
    
    mapping_fpath = os.path.join(output_dirpath, mapping_fname)
    with open(mapping_fpath, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Graph_ID", "Graph_File", "Subgraph_Files(multicol)"])
        writer.writerows(subgraph_file_mapping)

def _create_torch_subgraph(igraph: OmicsGraph, ofpath: str):

    node_features = np.zeros((len(igraph.vertices), len(igraph.markers)), dtype=np.float64)
    for vidx, vertex in enumerate(igraph.vertices):
        marker_data = np.array([igraph.node_attributes[vertex][mi] for mi in igraph.markers])
        node_features[vidx] = marker_data

    vertex_mapping = {vertex:vidx for vidx, vertex in enumerate(igraph.vertices)}
    
    edges = set(igraph.graph.edges())
    edge_index = np.zeros((2, len(edges) * 2), dtype=int)
    for eidx, (ei, ej) in enumerate(igraph.graph.edges()):
        ei_m, ej_m = vertex_mapping[ei], vertex_mapping[ej]
        edge_index[:, 2 * eidx] = [ei_m, ej_m]
        edge_index[:, 2 * eidx + 1] = [ej_m, ei_m]
    
    label = igraph.graph_label

    subgraph = Data(x = torch.tensor(node_features, dtype=torch.float), 
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            y = torch.tensor([label], dtype=torch.long))
    torch.save(subgraph, ofpath)

def create_torch_subgraphs(output_dirpath: str, mapping_fname: str = "torch_subgraph_mapping.csv", 
        py_subgraphs_dirname: str = "graphs_py", torch_subgraphs_dirname: str = "graphs_torch"):
    
    file_mapping = list()
    torchgraphs_output_dirpath = os.path.join(output_dirpath, torch_subgraphs_dirname)
    if not os.path.exists(torchgraphs_output_dirpath):
        os.makedirs(torchgraphs_output_dirpath)
    
    py_subgraphs_dirpath = os.path.join(output_dirpath, py_subgraphs_dirname)
    for ifile in os.listdir(py_subgraphs_dirpath):
        input_omicsgraph_fpath = os.path.join(py_subgraphs_dirpath, ifile)
        input_omicsgraph = load_omicsgraph(input_omicsgraph_fpath)
        
        output_fname = f'{os.path.splitext(ifile)[0]}.pt'
        output_torchgraph_fpath = os.path.join(torchgraphs_output_dirpath, output_fname)
        _create_torch_subgraph(input_omicsgraph, output_torchgraph_fpath)
        
        rel_pygraph_fpath = os.path.join(py_subgraphs_dirname, ifile)
        rel_torch_fpath = os.path.join(torch_subgraphs_dirname, output_fname)
        file_mapping.append([rel_pygraph_fpath, rel_torch_fpath])
    
    mapping_fpath = os.path.join(output_dirpath, mapping_fname)
    with open(mapping_fpath, 'w+') as ofile:
        writer = csv.writer(ofile, delimiter=',')
        writer.writerow(["Input_OmicsGraph", "Output_TorchGraph"])
        writer.writerows(file_mapping)
