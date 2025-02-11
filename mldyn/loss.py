import torch
import torch.nn as nn


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, predictions, targets):
        # predictions: tensor of shape (batch_size, n_timesteps, n_particles, input_state_dimension)
        # targets: tensor of shape (batch_size, n_timesteps, n_particles, input_state_dimension)
        return self.mse(predictions, targets)
