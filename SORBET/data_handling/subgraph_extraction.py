import numpy as np

from .graph_model import OmicsGraph

def subgraph_extraction(tissue_graph: OmicsGraph, extraction_method: str, extraction_args: dict = dict(), tissue_graph_for_feature_extraction: OmicsGraph = None):
    """Implements a basic interface to access different subgraph extraction methods (below) 
    """
    assert extraction_method in subgraph_extraction_methods
    return subgraph_extraction_methods[extraction_method](tissue_graph, **extraction_args, tissue_graph_for_feature_extraction=tissue_graph_for_feature_extraction)

def _microenvironment_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int, tissue_graph_for_feature_extraction: OmicsGraph = None):
    """Subgraph extraction algorithm
    """
    subgraphs = list()
    
    vertices, marker_values = tissue_graph.get_marker(marker)
    T = np.median(marker_values)
    Q = [vertices[i] for i in np.argsort(marker_values)]
    m = Q.pop()
        
    while tissue_graph.get_marker(marker, [m])[1][0] >= T: # TODO: Very ugly. Fix.
        neighborhood_1 = list(tissue_graph.get_khop_neighborhood(m, 1))
        _, marker_values = tissue_graph.get_marker(marker, neighborhood_1)
        
        if np.median(marker_values) >= T:
            neighborhood_k = list(tissue_graph.get_khop_neighborhood(m, k))
            
            if len(neighborhood_k) > minimum_size:
                neighborhood_k.append(m)
                if tissue_graph_for_feature_extraction is not None:
                    subgraph = tissue_graph_for_feature_extraction.make_subgraph(neighborhood_k)
                else:
                    subgraph = tissue_graph.make_subgraph(neighborhood_k)
                subgraphs.append(subgraph)

                Q = [vi for vi in Q if vi not in neighborhood_k]

        if len(Q) == 0: break
        
        m = Q.pop()
    
    return subgraphs

def _heat_diffusion_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int, tissue_graph_for_feature_extraction: OmicsGraph = None):
    """Subgraph extraction via heat diffusion-like process.
    """
    # TODO: Extract subgraphs using heat-diffusion like graph cover

    return [tissue_graph]

def _arbitrary_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int, tissue_graph_for_feature_extraction: OmicsGraph = None):
    """Subgraph extraction via an arbitrary selection of (minimally-overlapping) subgraphs 
    """

    subgraphs = list()
    vertices, marker_values = tissue_graph.get_marker(marker)
    Q = list(vertices)
    seen = set()
    
    pop_item = lambda v, Q: [vi for vi in Q if vi != v]

    while len(Q) > 0: 
        Vi = Q[np.random.choice(np.arange(len(Q)))]
        Q = pop_item(Vi, Q)
        
        if len(seen.intersection(tissue_graph.get_khop_neighborhood(Vi, 1))) != 0:
            continue

        neighborhood_k = list(tissue_graph.get_khop_neighborhood(Vi, k))
        if len(neighborhood_k) >= minimum_size: 
            seen |= set(neighborhood_k)
            
            subgraph = tissue_graph.make_subgraph(neighborhood_k)
            subgraphs.append(subgraph)

    return subgraphs


subgraph_extraction_methods = {
        "microenvironment": _microenvironment_subgraph_extraction,
        "arbitrary": _arbitrary_subgraph_extraction,
        "heat_diffusion": _heat_diffusion_subgraph_extraction
        }
