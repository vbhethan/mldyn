import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, particle_identities_path, window_size=20):
        # Change: data file now contains the "tapes" of independent dynamics runs
        # TODO: TimeSeriesDataset should construct the valid starting indices; the dataloader should iterate over those

        # Load trajectory "tapes" (independent dynamic runs)
        trajectory_tape_files = glob(os.path.join(data_path, "*.npy"))
        trajectory_tapes = [np.load(tape) for tape in trajectory_tape_files]
        self.tape_lengths = [tape.shape[0] for tape in trajectory_tapes]
        self.window_size = window_size

        self.concatenated_trajectory = np.concatenate(trajectory_tapes)
        # Valid starting indices are one such that you can step forward window_size steps
        # in the future without crossing into a new tape
        self.valid_starting_indices = []
        end_index = 0
        for tape_length in self.tape_lengths:
            self.valid_starting_indices.append(
                np.arange(end_index, end_index + tape_length - window_size)
            )
            end_index += tape_length
        self.valid_starting_indices = np.concatenate(self.valid_starting_indices)

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
        return self.valid_starting_indices.shape[0]

    def __getitem__(self, index):
        # Now we grab the starting index from the list of valid starting indices
        starting_index = self.valid_starting_indices[index]

        # Now we grab the trajectory window
        trajectory_window = self.concatenated_trajectory[
            starting_index : starting_index + self.window_size
        ]

        # The initial state is the first time step
        initial_condition = torch.FloatTensor(trajectory_window[0])

        # The targets are the subsequent time steps
        targets = torch.FloatTensor(trajectory_window[1:])

        return initial_condition, targets, self.particle_labels


def create_dataloader(
    data_path, particle_identities_path, batch_size, window_size=20, shuffle=True
):
    dataset = TimeSeriesDataset(
        data_path, particle_identities_path, window_size=window_size
    )
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
