import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mldyn.models.transformer import TransformerTimeSeriesModel
from mldyn.data.dataloaders import create_dataloader
from mldyn.loss import CustomMSELoss


def train(model, dataloader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        initial_condition, targets, particle_labels = batch
        initial_condition = initial_condition.to(device)
        targets = targets.to(device)
        particle_labels = particle_labels.to(device)

        optimizer.zero_grad()

        predictions = model(initial_condition, particle_labels)

        loss = criterion(predictions, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            initial_condition, targets, particle_labels = batch
            initial_condition = initial_condition.to(device)
            targets = targets.to(device)
            particle_labels = particle_labels.to(device)

            predictions = model(initial_condition, particle_labels)

            loss = sum(
                [criterion(pred, target) for pred, target in zip(predictions, targets)]
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    # Define hyperparameters
    n_particles = 148
    input_state_dimension = 6
    d_model = 128
    n_particle_types = 20
    n_time_steps = 20
    d_feedforward = 256

    num_epochs = 100
    learning_rate = 1e-4
    clip_value = 1.0
    batch_size = 4

    # Paths to datafiles
    train_data_path = "./train_data.npy"
    particle_identities_path = "particle_identities.txt"
    val_data_path = "./val_data.npy"

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = TransformerTimeSeriesModel(
        n_particles,
        input_state_dimension,
        d_model,
        n_particle_types,
        n_time_steps,
        d_feedforward,
    ).to(device)

    train_dataloader = create_dataloader(
        train_data_path, particle_identities_path, batch_size
    )

    criterion = CustomMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(
            model, train_dataloader, optimizer, criterion, device, clip_value
        )

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
