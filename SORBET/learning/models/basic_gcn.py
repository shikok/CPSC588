from typing import List, Dict, Union, Optional
from enum import Enum
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Linear, Sigmoid, ModuleList, Dropout, ReLU, BatchNorm1d, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, MeanAggregation, pool 

from .model_specification import StructureSpecification
from .base import BaseGraphModel 
from ._utils import parse_pooling_function

_BasicGCNSpecification = StructureSpecification(
        model_serializable_parameters = ["conv_channels", "out_channels"],
        model_non_serializable_parameters = [("pooling_fn", 'choice', False), ("dropout", float, False), ("l1_penalty", float, True)],
        model_input_specifier = "in_channel",
        random_seed_specifier = "random_seed"
        )

class GCNStandardSupervised(BaseGraphModel):

    def __init__(self, 
            in_channel: int,
            conv_channels: List[int],
            out_channels: List[int],
            dropout: float = 0.5, batch_norm_args: dict = {'eps': 1e-5, 'momentum': 0.1}, 
            pooling_fn: pool = 'global_mean',
            loss_fn: torch.nn = BCEWithLogitsLoss, l1_penalty: Union[float, None] = None,
            random_seed: Optional[int] = None
            ):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Convolutional layers:
        self.conv_layers = ModuleList()
        self.pass_edge_idx = list()
        for in_channel, out_channel in zip([in_channel, *conv_channels], conv_channels):
            self.conv_layers.extend([
                Dropout(p = dropout), 
                GCNConv(in_channel, out_channel),
                BatchNorm1d(out_channel, **batch_norm_args),
                ReLU()
                ])
            self.pass_edge_idx.extend([False, True, False, False])

        self.pooler = parse_pooling_function(pooling_fn) 
        
        self.lin_layers = ModuleList()
        for in_channel, out_channel in zip([conv_channels[-1], *out_channels], [*out_channels, 1]):
            self.lin_layers.extend([
                Dropout(p = dropout),
                Linear(in_channel, out_channel),
                BatchNorm1d(out_channel, **batch_norm_args),
                ReLU()
                ])
        self.lin_layers = self.lin_layers[:-1] # Remove final ReLU

        self.output_nonlin = Sigmoid()
        
        # Initialize loss function
        self.loss_function = loss_fn()
        self.l1_penalty = l1_penalty if l1_penalty is not None else 0

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.get_cell_embedding(x, edge_index, batch)
        x = self.get_subgraph_embedding(x, edge_index, batch)
        
        for op in self.lin_layers:
            x = op(x)
        
        return x

    def predict(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.forward(x, edge_index, batch)
        return self.output_nonlin(x)
   
    def get_loss_function(self, predictions: Tensor, labels: Tensor) -> Tensor: 
        loss = self.loss_function(predictions, labels)

        if self.l1_penalty > 0:  
            for p in self.parameters():
                loss += self.l1_penalty * torch.linalg.norm(torch.flatten(p), 1)
        
        return loss
    
    def get_cell_embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Compute graph convolutions
        for op, pass_ei in zip(self.conv_layers, self.pass_edge_idx):
            if pass_ei:
                x = op(x, edge_index)
            else:
                x = op(x)
        
        return x
    
    def get_subgraph_embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Pools over all cell embeddings
        # TODO: Input currently after cell embedding. Should cell embedding be integrated into the function?
        return self.pooler(x, batch)

    @staticmethod
    def get_model_specification():
        return _BasicGCNSpecification
