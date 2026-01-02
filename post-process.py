import torch
import numpy as np
from tqdm import tqdm

from mldyn.postprocessing.transformer_postprocess_functions import (
    load_trained_model,
    prepare_input_data,
    extract_attention_maps,
)
from mldyn.models.transformer import TransformerTimeSeriesModel


def main(data_path, particle_identities_path, model_path):
    # Hyperparameters aligned with train.py
    n_particles = 20
    input_state_dimension = 6
    d_model = 128
    n_particle_types = 20
    n_time_steps = 19  # window_size=20 -> 19 target steps
    d_feedforward = 256

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with attention map output enabled
    model = TransformerTimeSeriesModel(
        n_particles=n_particles,
        input_state_dimension=input_state_dimension,
        d_model=d_model,
        n_particle_types=n_particle_types,
        n_time_steps=n_time_steps,
        d_feedforward=d_feedforward,
        return_attention=True,  # Enable attention map output
    ).to(device)

    # Load trained weights
    model = load_trained_model(model_path, model)

    # Prepare input data
    dataset = prepare_input_data(data_path, particle_identities_path)

    # Initialize running sums for averaging attention maps
    running_sum = None
    sample_count = 0

    # Process each sample
    for data in tqdm(dataset):
        ic, _, pl = data
        # Add batch dimension
        ic = ic.unsqueeze(0).to(device)
        pl = pl.unsqueeze(0).to(device)

        # Extract attention maps
        encoder_maps, decoder_self_maps, decoder_cross_maps = extract_attention_maps(
            model, ic, pl
        )

        # Combine attention weights for this sample
        combined_attention = encoder_maps + decoder_self_maps + decoder_cross_maps

        # Update running sum
        if running_sum is None:
            running_sum = combined_attention
        else:
            running_sum += combined_attention

        sample_count += 1

        # Clean up GPU memory
        del encoder_maps, decoder_self_maps, decoder_cross_maps, combined_attention
        torch.cuda.empty_cache()

    # Compute final average
    average_attention_weights = running_sum / sample_count

    # Save the averaged attention weights
    np.save("combined_attention_weights.npy", average_attention_weights)


if __name__ == "__main__":
    # Data Paths consistent with training defaults
    model_path = "./model.pth"
    data_path = "./sim_data/"
    particle_identities_path = "./sim_data/particle_identities.txt"

    main(
        data_path=data_path,
        particle_identities_path=particle_identities_path,
        model_path=model_path,
    )

    print("Attention maps computed and saved to disk successfully")
