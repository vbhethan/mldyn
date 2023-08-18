import os
import h5py
from torch.utils.data import Dataset, Dataloader
import torch


class DynamicsDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = h5py.File(hdf5_file, "r")
        self.transform = transform

        # TODO: assume the hdf5 file has been processed to have examples; get some metadata like the number of examples
        # Getting items should involve grabbing the dataset with the specified index from the hdf5 file

        with h5py.File(self.hdf5_file) as f:
            self.num_examples = len(f["examples"])

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with h5py.File(self.hdf5_file) as f:
            coordinates = torch.Tensor(f["examples"][f"{idx}"]["coordinates"][:])
            velocities = torch.Tensor(f["examples"][f"{idx}"]["velocities"][:])

        sample = {"coordinates": coordinates, "velocities": velocities}

        return sample
