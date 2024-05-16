from pathlib import Path
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

# from torch_geometric.data import Data


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


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    encoded_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return encoded_labels


def graph_topology_from_pdb(path_to_pdb):
    """generates a graph toplogy from a pdb file

    Args:
        path_to_pdb (str): path to the pdb file

    Returns:
        x: (N_nodes,dim_embedding) torch.Tensor
        edge_index: (2,N_edges) torch.Tensor, edge indices in form expected by pytorch_geo
        edge_attr: (N_edges, 1) torch.Tensor, edge embeddings of pair-wise distances

    """

    restypes, coordinates = read_pdb(path_to_pdb)
    dist_matrix = squareform(pdist(coordinates))

    # make a list of one-hot vectors for each residue
    res_encodings = encode_onehot(restypes)

    edge_index = []
    edge_attr = []
    x = torch.zeros(
        (len(restypes), res_encodings.shape[1] + 3), dtype=torch.float
    )  # For now, assuming only positions (encoding vector length + 3 position dims) TODO: detect if velocities available and change hardcoded dimension
    for i in range(res_encodings):
        # Add positional data to nodes (i.e., position, for now, will add in velocities later)
        x[i] = torch.cat([res_encodings[i], coordinates[i, :]], dim=0)
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
