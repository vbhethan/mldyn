import torch
from pathlib import Path
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import pdist, squareform


def read_pdb(path_to_pdb):
    restypes = []
    coordinates = []
    pdb_lines = Path(path_to_pdb).read_text().split("\n")
    # Only keep lines that start with "ATOM"
    pdb_lines = [line for line in pdb_lines if line.startswith("ATOM")]
    for line in pdb_lines:
        restypes.append(line[17:20])
        coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    coordinates = torch.Tensor(coordinates)
    return restypes, coordinates


restype_one_hot_encoding = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}

for key, value in restype_one_hot_encoding.items():
    one_hot_vec = np.zeros(20)
    one_hot_vec[value] = 1
    restype_one_hot_encoding[key] = one_hot_vec


def graph_topology_from_pdb(path_to_pdb):
    restypes, coordinates = read_pdb(path_to_pdb)
    dist_matrix = squareform(pdist(coordinates))

    # make a list of one-hot vectors for each residue
    res_encoding = [restype_one_hot_encoding[restype] for restype in restypes]
    res_encoding = torch.tensor(res_encoding, dtype=torch.float)

    edge_index = []
    edge_attr = []
    x = torch.zeros(
        (len(restypes), 23), dtype=torch.float
    )  # For now, assuming only positions TODO: detect if velocities available and change hardcoded dimension
    for i in range(len(restypes)):
        # Add positional data to nodes (i.e., position, for now, will add in velocities later)
        x[i] = torch.cat([res_encoding[i], coordinates[i, :]], dim=0)
        for j in range(i + 1, len(restypes)):
            # Add pairwise distance to edges, bidirectional
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(dist_matrix[i, j])
            edge_attr.append(dist_matrix[j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Need to unsqueeze to get the right shape for PyTorch Geometric (edge_attr should be [N_edges, edge_dim])
    edge_attr = edge_attr.unsqueeze(1)

    return x, edge_index, edge_attr


def update_graph_node_attributes(node_one_hot, new_node_attr):
    x = torch.cat([node_one_hot, new_node_attr])
    return x


def append_node_data_to_graph(x, edge_index, edge_attr, position_matrix):
    # use the old graph topology but update the edge attributes to reflect distances from the new position matrix
    distance_matrix = squareform(pdist(position_matrix))

    for ind, (src, dst) in enumerate(edge_index.t().tolist()):
        edge_attr[ind] = distance_matrix[src, dst]

    return x, edge_index, edge_attr
