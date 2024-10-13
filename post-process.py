import torch
import numpy as np
import os

from mldyn.postprocessing.transformer_postprocess_functions import (
    load_trained_model,
    prepare_input_data,
    extract_attention_maps,
    combine_attention_weights,
)
from mldyn.postprocessing.transformer_postprocess_model import (
    PostProcessTransformerTimeSeriesModel,
)


def main(data_path, particle_identities_path, model_path):
    # Define hyperparameters (Make sure these are the same as the training script)
    n_particles = 148
    input_state_dimension = 6
    d_model = 128
    n_particle_types = 20
    n_time_steps = 19
    d_feedforward = 256

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = PostProcessTransformerTimeSeriesModel(
        n_particles,
        input_state_dimension,
        d_model,
        n_particle_types,
        n_time_steps,
        d_feedforward,
    ).to(device)

    model = load_trained_model(model_path, model)

    # Prepare input data
    dataset = prepare_input_data(data_path, particle_identities_path)

    all_encoder_attention_maps = []
    all_decoder_self_attention_maps = []
    all_decoder_cross_attention_maps = []

    # Extract attention maps
    for data in dataset:
        ic, _, pl = data
        # Add a batch dimension, since the model expects a batch
        ic = ic.unsqueeze(0)
        pl = pl.unsqueeze(0)

        (
            encoder_attention_maps,
            decoder_self_attention_maps,
            decoder_cross_attention_maps,
        ) = extract_attention_maps(model, ic, pl)

        all_encoder_attention_maps.append(encoder_attention_maps)
        all_decoder_self_attention_maps.append(decoder_self_attention_maps)
        all_decoder_cross_attention_maps.append(decoder_cross_attention_maps)

    # Combine attention weights
    combined_attention_weights = combine_attention_weights(
        encoder_attention_maps,
        decoder_self_attention_maps,
        decoder_cross_attention_maps,
    )

    # The combined attention weights will have shape (n_samples, t_steps, n_particles, n_particles)
    # We will aggregate the attention maps across the sample dimension
    combined_attention_weights = np.mean(combined_attention_weights, axis=0)
    # Now the combined attention weights have shape (t_steps, n_particles, n_particles)

    # Save the combined attention weights
    np.save("combined_attention_weights.npy", combined_attention_weights)


if __name__ == "__main__":

    # Data Paths
    model_path = "./working_experiments/CAM-system/trained-model-1/CAM-model.pth"
    data_path = "./working_experiments/CAM-system/trained-model-1/subsample_100.npy"
    particle_identities_path = (
        "./working_experiments/CAM-system/trained-model-1/particle_identities.txt"
    )

    main(
        data_path=data_path,
        particle_identities_path=particle_identities_path,
        model_path=model_path,
    )
