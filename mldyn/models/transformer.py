import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from mldyn.layers.layers import (
    EncoderLayer,
    DecoderLayer,
    ResidueEmbeddingLayer,
)


class TransformerTimeSeriesModel(nn.Module):
    """
    Transformer model for time series data
    Specify the model with the arguments
    Args:
        - n_particles: Number of particles in the system
        # TODO: finish docstring
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
        num_heads: int = 4,
    ):
        super(TransformerTimeSeriesModel, self).__init__()
        self.n_time_steps = n_time_steps
        self.input_state_dimension = input_state_dimension
        self.residue_embedding = ResidueEmbeddingLayer(
            n_particles, input_state_dimension, d_model, n_particle_types
        )

        # Create separate encoder and decoder for each time step
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(d_model, d_feedforward, dropout, num_heads=num_heads)
                for _ in range(n_time_steps)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                DecoderLayer(d_model, d_feedforward, dropout, num_heads=num_heads)
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

        for t in range(self.n_time_steps):
            # Encode z from x
            z = self.encoders[t](x)

            decoder_input = (
                x if t == 0 else self.residue_embedding(predictions[-1], particle_types)
            )

            # z is the memory, decoder_input is what will attend to it
            output = self.decoders[t](decoder_input, z)

            # This output is what we're trying to predict
            predictions.append(output)

        return predictions


# Example usage
if __name__ == "__main__":
    n_particles = 100
    input_state_dimension = 6
    d_model = 128
    n_particle_types = 20
    n_time_steps = 19
    d_feedforward = 256
    batch_size = 4

    model = TransformerTimeSeriesModel(
        n_particles,
        input_state_dimension,
        d_model,
        n_particle_types,
        n_time_steps,
        d_feedforward,
    )

    print(
        f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}"
    )

    initial_condition = torch.randn(batch_size, n_particles, input_state_dimension)
    particle_types = torch.randint(0, n_particle_types, (batch_size, n_particles))

    predictions = model(initial_condition, particle_types)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Shape of each prediction: {predictions[0].shape}")
