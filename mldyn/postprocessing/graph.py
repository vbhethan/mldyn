"""
Module for post-processing trained models to graph data structures and visualizations.
"""

import os
import torch
import numpy as np

from mldyn.data.dataloaders import load_data
from mldyn.utils import encode_onehot


def load_model(model, state_dict_path, device="cpu"):
    """Load a trained model."""
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_edge_logits(model_path, num_atoms, data_loader, device="cpu"):
    """Get predicted edges from a trained model."""
    # TODO: you can probably just get the num_atoms from the data_loader...
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

    rel_rec = torch.from_numpy(encode_onehot(np.where(off_diag)[0])).float()
    rel_send = torch.from_numpy(encode_onehot(np.where(off_diag)[1])).float()

    device = torch.device(device)
    model = torch.load(os.path.join(model_path))
    model.eval()
    predicted_edges = []
    with torch.no_grad():
        for data in data_loader:
            example = torch.stack(data).to(device)
            example = example.squeeze(0)
            logits = model(example, rel_rec, rel_send).cpu().numpy()

            predicted_edges.append(logits)
    return np.concatenate(predicted_edges), rel_rec, rel_send


def get_predicted_adjacency_matrix(
    model_path, data_path, num_atoms, batch_size=1, device="cpu"
):
    """Get predicted edges from a trained model."""
    data_loader, _, _ = load_data(data_path, batch_size=batch_size)
    logits, rel_rec, rel_send = predict_edge_logits(
        model_path, num_atoms, data_loader, device
    )
    receivers = torch.nonzero(rel_rec)[:, 1]
    senders = torch.nonzero(rel_send)[:, 1]
    num_edge_types = logits.shape[-1]
    adjacency_matrix = np.zeros((logits.shape[0], num_atoms, num_atoms, num_edge_types))
    for i, logit in enumerate(logits):
        for value, rec, send in zip(logit, receivers, senders):
            adjacency_matrix[i, int(rec), int(send)] = value
    return adjacency_matrix


def get_predicted_edges(model_path, data_path, num_atoms, batch_size=1, device="cpu"):
    """
    Get predicted edges and return in COO format
    """
    data_loader, _, _ = load_data(data_path, batch_size=batch_size)
    predicted_edges, rel_rec, rel_send = predict_edge_logits(
        model_path, num_atoms, data_loader, device
    )
    receivers = torch.nonzero(rel_rec)[:, 1]
    senders = torch.nonzero(rel_send)[:, 1]
    return predicted_edges, receivers, senders
