import torch
from torch_geometric.nn import MessagePassing


class Node2Edge(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, edge_hidden_dim, edge_out_dim):
        super().__init__(aggr="add")
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_in_dim + edge_in_dim, edge_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden_dim, edge_out_dim),
        )

    def forward(self, edge_index, x, edge_attr):
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        return edge_attr

    def edge_update(self, x_j, x_i, edge_attr):
        # pytorch geometric defines messages as going from node x_j to node x_i TODO: double check this definition...
        out = torch.cat([x_j, x_i, edge_attr], dim=1)
        out = self.mlp(out)
        return out


class Edge2Node(MessagePassing):
    # TODO: need to eventually make a version of this that can handle multiple edge types? Have a different MLP for each edge type?
    # (And make one edge type 0 explicitly)
    def __init__(self, edge_in_dim, node_in_dim, node_hidden_dim, node_out_dim):
        super().__init__(aggr="add")
        self.edge_in_dim = edge_in_dim
        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_in_dim + edge_in_dim, node_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_hidden_dim, node_out_dim),
        )

    def forward(self, edge_index, x, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        out = torch.cat([x_j, x_i, edge_attr], dim=1)
        out = self.mlp(out)
        return out
