from typing import List
import numpy as np

from anndata import AnnData
import scanpy as sc

from .graph_model import OmicsGraph

def normalize_graph(tissue_graph: OmicsGraph, normalize_method: str, normalize_args: dict = dict()):
    """Normalizes for a single sample. Shallow wrapper around _normalize_method
    """
    assert normalize_method in normalization_method_fns
  
    data = tissue_graph.get_node_data()
    normed_data = normalization_method_fns[normalize_method](data, normalize_method, **normalize_args) 
    tissue_graph.set_node_data(normed_data) 

def normalize_dataset(tissue_graphs: List[OmicsGraph], normalize_method: str, normalize_args: dict = dict()):
    """Normalizes across multiple samples. Shallow wrapper around _normalize_method
    """
    assert normalize_method in normalization_method_fns

    # Serialize data into single matrix:
    data, data_indices = list(), list()
    last_index = 0 
    for graph in tissue_graphs:
        graph_data = graph.get_node_data()
        
        sidx = last_index
        last_index = sidx + graph_data.shape[0]

        data_indices.append((sidx, last_index))
        data.append(graph_data)
    
    data = np.vstack(data)
    normed_data = normalization_method_fns[normalize_method](data, normalize_method, **normalize_args)
    
    # De-serialize data matrix:
    for tissue_graph, (sidx, eidx) in zip(tissue_graphs, data_indices):
        sample_normed_data = normed_data[sidx:eidx]
        tissue_graph.set_node_data(sample_normed_data)

def _normalize_by_total_count(data: np.ndarray, normalize_method: str, normalize_total_args: dict = dict()):
    if normalize_method ==  'log_normalize':
        data = sc.pp.log1p(data, copy=True)

    adata = AnnData(data)
    
    default_args = {
            'exclude_highly_expressed': True,
            'max_fraction': 0.05,
            'target_sum': 1.0
            }
    default_args.update(normalize_total_args)
    default_args['inplace'] = False 
   
    normed_data = sc.pp.normalize_total(adata, **default_args)['X']
    return normed_data

def _normalize_by_log(data: np.ndarray, normalize_method: str, normalize_log_args: dict = dict()):
    normed_data = sc.pp.log1p(data, copy=True)
    
    if 'rescale' in normalize_log_args:
        normed_data = normed_data / normalize_log_args['rescale']
    
    return normed_data

def _normalize_by_pca(data: np.ndarray, normalize_method: str, pca_args: dict = dict(), fraction_variance_explained: float = 0.9): 
    adata = AnnData(data)

    default_args = {}
    default_args.update(pca_args)
    default_args['inplace'] = False

    pca_results = sc.pp.pca(adata, **default_args)
    var = pca_results.uns['pca']['variance_ratio']
    
    idx = np.min(np.argwhere(np.cumsum(var) >= fraction_variance_explained))
    normed_data = pca_results.obsm['X_pca'][:,:idx]

    return normed_data

def _normalize_by_variance(data: np.ndarray, normalize_method: str, zero_center: bool = True): 
    """Scales data to unit variance and zero mean. 
    """
    return sc.pp.scale(data, zero_center = zero_center, copy = True)

def _normalize_to_range(data: np.ndarray, normalize_method: str):
    mi, ma = np.min(data, axis=0), np.max(data, axis=0)
    return (data - mi) / (ma - mi)

normalization_method_fns = {
        'total_count': _normalize_by_total_count,
        'log_normalize': _normalize_by_log,
        'pca': _normalize_by_pca,
        'z-normalize': _normalize_by_variance,
        'to-range': _normalize_to_range
        }

def normalize_graphs_by_pca(tissue_graphs: List[OmicsGraph], pca_args: dict = dict()):
    combined_data = list()
    offsets, cidx = [0], 0
    
    for graph in tissue_graphs:
        graph_data = graph.get_node_data()
        combined_data.append(graph_data)

        cidx += graph_data.shape[0]
        offsets.append(cidx)
    
    combined_data = np.vstack(combined_data)
    adata = AnnData(combined_data)

    default_args = {}
    default_args.update(pca_args)
    default_args['copy'] = True

    pca_results = sc.pp.pca(adata, **default_args)
    pca_data = pca_results.obsm['X_pca']
    
    mupd = ['PC {idx}' for idx in range(pca_data.shape[1])]
    for sidx, eidx, tissue_graph in zip(offsets, offsets[1:], tissue_graphs):
        normed_data = pca_data[sidx:eidx]
        tissue_graph.set_node_data(normed_data, marker_update = mupd) 
