import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

from mldyn.data.dataloaders import TimeSeriesDataset
from mldyn.postprocessing.transformer_postprocess_model import (
    PostProcessTransformerTimeSeriesModel,
)


def load_trained_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    return model


def prepare_input_data(data_path, particle_identities_path):
    dataset = TimeSeriesDataset(data_path, particle_identities_path)
    initial_conditions, _, particle_labels = dataset
    # Could use a batch or index sampling, but for postprocessing I think I will randomly sample initial conditions and store in a separate file
    initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
    return initial_conditions, particle_labels


def extract_attention_maps(model, initial_condition, particle_labels):
    model.eval()
    with torch.no_grad():
        (
            predictions,
            encoder_attention_maps,
            decoder_self_attention_maps,
            decoder_cross_attention_maps,
        ) = model(initial_condition, particle_labels)

    # The attention maps will have shape (t_steps, batch_size, n_particles, n_particles)
    # We will aggregate the attention maps across the batch dimension (for batch size 1 this just drops the batch dimension)
    encoder_attention_maps = np.mean(np.stack(encoder_attention_maps), axis=1)
    decoder_self_attention_maps = np.mean(np.stack(decoder_self_attention_maps), axis=1)
    decoder_cross_attention_maps = np.mean(
        np.stack(decoder_cross_attention_maps), axis=1
    )
    # Now the maps have shape (t_steps, n_particles, n_particles)

    return (
        encoder_attention_maps,
        decoder_self_attention_maps,
        decoder_cross_attention_maps,
    )


def visualize_attention_maps(
    encoder_attention_maps,
    decoder_self_attention_maps,
    decoder_cross_attention_maps,
    figure_directory="./",
):

    # Start with encoder attention maps
    for t, encoder_attention_map in enumerate(encoder_attention_maps):
        plt.figure(figsize=(10, 8))
        plt.imshow(encoder_attention_map.squeeze().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Encoder Attention Map at Time Step {t}")
        plt.savefig(os.path.join(figure_directory, f"encoder_attention_map_{t}.png"))
        plt.close()

    # Next, decoder self attention maps
    for t, decoder_self_attention_map in enumerate(decoder_self_attention_maps):
        plt.figure(figsize=(10, 8))
        plt.imshow(decoder_self_attention_map.squeeze().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Decoder Self Attention Map at Time Step {t}")
        plt.savefig(
            os.path.join(figure_directory, f"decoder_self_attention_map_{t}.png")
        )
        plt.close()

    # Finally, decoder cross attention maps
    for t, decoder_cross_attention_map in enumerate(decoder_cross_attention_maps):
        plt.figure(figsize=(10, 8))
        plt.imshow(decoder_cross_attention_map.squeeze().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"Decoder Cross Attention Map at Time Step {t}")
        plt.savefig(
            os.path.join(figure_directory, f"decoder_cross_attention_map_{t}.png")
        )
        plt.close()


def combine_attention_weights(
    encoder_attention, decoder_self_attention, decoder_cross_attention
):
    """
    Combine the attention weights from the encoder, decoder self attention, and decoder cross attention
    """
    # TODO: test different weights for each attention map?
    combined = encoder_attention + decoder_self_attention + decoder_cross_attention
    # Normalize so all attention weights sum to 1 for a target particle
    return combined / (combined.sum(axis=-1, keepdims=True))
