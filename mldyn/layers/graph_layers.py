import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size


class Node2Edge(torch.nn.Module):
    def __init__(
        self, n_node_features, n_edge_features, hidden_edge_features, out_edge_features
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_features + n_edge_features, hidden_edge_features),
            nn.ReLU(),
            nn.Linear(hidden_edge_features, out_edge_features),
        )

    def forward(self, src, dst, edge_attr):
        out = torch.cat([src, dst, edge_attr], 1)
        out = self.edge_mlp(out)

        return out
