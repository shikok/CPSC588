#from typing import *
import torch
from abc import abstractmethod

from .model_specification import StructureSpecification 

# This model structure is inspired by: https://github.com/AntixK/PyTorch-VAE

class BaseGraphModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor: 
        raise NotImplementedError

    @abstractmethod
    def get_loss_function(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor: 
        raise NotImplementedError

    @abstractmethod
    def get_cell_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_subgraph_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def get_model_specification() -> StructureSpecification:
        raise NotImplementedError
