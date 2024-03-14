import os
import h5py
from torch.utils.data import Dataset, DataLoader  
from torch.utils.data.dataset import TensorDataset
import torch
import numpy as np


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

        # TODO: assumes right now that you are only loading one example at a time, check if I need to account for a list of indices passed to the 
        # getitem method
        with h5py.File(self.hdf5_file) as f:
            coordinates = torch.Tensor(f["examples"][f"{idx}"]["coordinates"][:])
            velocities = torch.Tensor(f["examples"][f"{idx}"]["velocities"][:])

        sample = {"coordinates": coordinates, "velocities": velocities}

        return sample
    
def load_data(file_basename, batch_size=1, data_dir="./sim_data"):
    #NOTE: currently, hard-coding a 3-dimensional dataset, i.e. we only stored positions.
    #TODO: eventually, need to make this general to handle any number of dimensions
        loc_train = os.path.join(data_dir, "{}.npy".format(file_basename))
        #TODO: eventually, implement a train-valid-test split

        # Shape [num_samples, num_timesteps, num_atoms, num_dimensions]
        num_atoms = loc_train.shape[2]  # Fix: calculate the number of atoms correctly

        loc_max = loc_train.max()
        loc_min = loc_train.min()

       # Reshape to: [num_samples, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, (0, 2, 1, 3))

        # Normalize position data to be in the range [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1

        feat_train = torch.FloatTensor(loc_train)
        train_data = TensorDataset(feat_train)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)  
        
        return train_dataloader, loc_max, loc_min


