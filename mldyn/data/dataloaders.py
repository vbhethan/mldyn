import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, particle_identities_path):
        # Load the full time series data
        # Shape: (N_windows, T_steps_per_window, N_particles, particle_state_dimension)
        self.data = np.load(data_path)

        # Load particle identities
        with open(particle_identities_path, "r") as f:
            sequence_string = f.read().strip()
            particle_identities = [
                amino_acid_code for amino_acid_code in sequence_string
            ]

        # Convert particle identities to integer labels
        unique_identities = list(set(particle_identities))
        self.identity_to_label = {
            identity: i for i, identity in enumerate(unique_identities)
        }
        self.particle_labels = torch.tensor(
            [self.identity_to_label[identity] for identity in particle_identities]
        )

    def __len__(self):
        # Return the number of available samples (windows)
        return self.data.shape[0]

    def __getitem__(self, index):
        # Get the trajectory window
        trajectory_window = torch.FloatTensor(self.data[index])

        # The initial state is the first time step
        initial_condition = trajectory_window[0]

        # The targets are the subsequent time steps
        targets = trajectory_window[1:]

        return initial_condition, targets, self.particle_labels


def create_dataloader(data_path, particle_identities_path, batch_size, shuffle=True):
    dataset = TimeSeriesDataset(data_path, particle_identities_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Example usage
if __name__ == "__main__":
    data_path = "./example_data.npy"
    particle_identities_path = "particle_id_rand.txt"
    batch_size = 4

    dataloader = create_dataloader(data_path, particle_identities_path, batch_size)

    for batch in dataloader:
        initial_condition, targets, particle_labels = batch
        print(f"Input conditions shape: {initial_condition.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Particle labels shape: {particle_labels.shape}")
        break
