import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from mldyn.postprocessing.transformer_postprocess_layers import (
    PostProcessEncoderLayer,
    PostProcessDecoderLayer,
    ResidueEmbeddingLayer,
)


class PostProcessTransformerTimeSeriesModel(nn.Module):
    """
    Transformer model for time series data
    Specify the model with the arguments
    Modified to report attention weights
    (see mldyn.models.transformer.TransformerTimeSeriesModel)
    """

    def __init__(
        self,
        n_particles: int,
        input_state_dimension: int,
        d_model: int,
        n_particle_types: int,
        n_time_steps: int,
        d_feedforward: int,
        dropout: float = 0.0,
    ):
        super(PostProcessTransformerTimeSeriesModel, self).__init__()
        self.n_time_steps = n_time_steps
        self.input_state_dimension = input_state_dimension
        self.residue_embedding = ResidueEmbeddingLayer(
            n_particles, input_state_dimension, d_model, n_particle_types
        )

        # Create separate encoder and decoder for each time step
        self.encoders = nn.ModuleList(
            [
                PostProcessEncoderLayer(d_model, d_feedforward, dropout)
                for _ in range(n_time_steps)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                PostProcessDecoderLayer(d_model, d_feedforward, dropout)
                for _ in range(n_time_steps)
            ]
        )

        self.final_layers = nn.ModuleList(
            [nn.Linear(d_model, input_state_dimension) for _ in range(n_time_steps)]
        )

    def forward(
        self, initial_condition: torch.Tensor, particle_types: torch.Tensor
    ) -> List[torch.Tensor]:
        # initial_condition shape: (batch_size, n_particles, input_state_dimension)
        # particle_types shape: (batch_size, n_particles)

        # Embed the initial condition
        x = self.residue_embedding(initial_condition, particle_types)

        predictions = []

        # # intialize z to be equal to embedded initial condition
        # z = x

        # Initialize lists to store attention weights
        encoder_attention_maps = []
        decoder_self_attention_maps = []
        decoder_cross_attention_maps = []

        for t in range(self.n_time_steps):
            # Encode z from x
            z, encoder_attention = self.encoders[t](x)
            # Store the attention weight for this timestep encoder
            encoder_attention_maps.append(encoder_attention)

            # For the first step use the initial condition as the target
            # For subsequent steps use the previous prediction
            target = (
                x if t == 0 else self.residue_embedding(predictions[-1], particle_types)
            )

            # Decode
            output, decoder_self_attention, decoder_cross_attention = self.decoders[t](
                z, target
            )
            # Store the attention weights for this timestep decoder
            decoder_self_attention_maps.append(decoder_self_attention)
            decoder_cross_attention_maps.append(decoder_cross_attention)

            # Project to the input state dimension
            prediction = self.final_layers[t](output)

            predictions.append(prediction)

        return (
            predictions,
            encoder_attention_maps,
            decoder_self_attention_maps,
            decoder_cross_attention_maps,
        )
