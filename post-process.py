import torch
import numpy as np
from tqdm import tqdm

from mldyn.postprocessing.transformer_postprocess_functions import (
    load_trained_model,
    prepare_input_data,
    extract_attention_maps,
)
from mldyn.postprocessing.transformer_postprocess_model import (
    PostProcessTransformerTimeSeriesModel,
)

def main(data_path, particle_identities_path, model_path):
    # Define hyperparameters (Make sure these are the same as the training script)
    n_particles = 388
    input_state_dimension = 6
    d_model = 128
    n_particle_types = 20
    n_time_steps = 49
    d_feedforward = 128

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
    dataset = prepare_input_data(data_path, particle_identities_path, device)

    # Initialize running sums and counts
    running_sum = None
    sample_count = 0

    # Extract attention maps and compute running average
    for data in tqdm(dataset):
        ic, _, pl = data
        # Add a batch dimension, since the model expects a batch
        ic = ic.unsqueeze(0)
        pl = pl.unsqueeze(0)

        (
            encoder_attention_maps,
            decoder_self_attention_maps,
            decoder_cross_attention_maps,
        ) = extract_attention_maps(model, ic, pl)

        # Combine attention weights for this sample
        combined_attention = (
            encoder_attention_maps +
            decoder_self_attention_maps +
            decoder_cross_attention_maps
        )

        # Update running sum
        if running_sum is None:
            running_sum = combined_attention
        else:
            running_sum += combined_attention

        sample_count += 1

        del encoder_attention_maps, decoder_self_attention_maps, decoder_cross_attention_maps, combined_attention

    # Compute final average
    combined_attention_weights = running_sum / sample_count

    # Save the combined attention weights
    np.save("combined_attention_weights.npy", combined_attention_weights)

if __name__ == "__main__":
    # Data Paths
    model_path = "./model.pth"
    data_path = "../sim_data/subselect_2000.npy"
    particle_identities_path = "../particle_identities.txt"

    main(
        data_path=data_path,
        particle_identities_path=particle_identities_path,
        model_path=model_path,
    )

    print("Attention maps computed and saved to disk sucessfully")

    
