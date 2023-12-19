import os
import csv
from dataclasses import dataclass

import torch
from torch_geometric.data import Dataset

class TorchOmicsDataset(Dataset):
    def __init__(self, root, subgraph_metadata, split=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self._processed_dir = subgraph_metadata.processed_dirpath
        self._processed_files = _load_filtered_graphs(split, root, subgraph_metadata.subgraph_map, subgraph_metadata.torch_subgraph_map)

    @property
    def processed_dir(self):
        return self._processed_dir 
    
    @property
    def processed_file_names(self):
        return self._processed_files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

@dataclass
class SubgraphMetadata:
    processed_dirpath: str
    subgraph_map: str
    torch_subgraph_map: str

def make_subgraph_metadata(processed_dirpath: str, py_mapping_fname: str = "py_subgraph_mapping.csv", torch_mapping_fname: str = "torch_subgraph_mapping.csv"):
    py_map_fpath = os.path.join(processed_dirpath, py_mapping_fname)
    torch_map_fpath = os.path.join(processed_dirpath, torch_mapping_fname)

    assert os.path.exists(py_map_fpath)
    assert os.path.exists(torch_map_fpath)
    
    return SubgraphMetadata(os.path.abspath(processed_dirpath), py_map_fpath, torch_map_fpath)

def _load_filtered_graphs(included_graphs, datadir, subgraph_mapping_fpath, torchgraph_mapping_fpath):
    """Finds mapped torch filepaths 
    """
    if included_graphs is None or len(included_graphs) == 0:
        with open(os.path.join(datadir, torchgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            torchgraphs = [t[1] for t in reader]
    else:
        subgraphs = list()
        with open(os.path.join(datadir, subgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            for row in filter(lambda t: t[0] in included_graphs, reader):
                subgraphs.extend(row[2:])

        with open(os.path.join(datadir, torchgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            torchgraphs = [t[1] for t in reader if t[0] in subgraphs]
    
    return torchgraphs
