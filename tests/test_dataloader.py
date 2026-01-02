import pytest
import numpy as np
import torch
import tempfile
import os
from mldyn.data.dataloaders import TimeSeriesDataset, create_dataloader


@pytest.fixture
def sample_data():
    # Create temporary test data
    n_tapes = 2  # Number of independent trajectory tapes
    tape_length = 10  # Length of each tape
    time_steps = tape_length  # Total timesteps in each tape
    n_particles = 3
    state_dim = 6  # e.g., x, y, z coordinates + r sidechain
    window_size = 5  # number of steps in the window

    # Calculate number of possible windows
    n_windows = (time_steps - window_size) * n_tapes  # tapes are equal sized,
    # TODO: write test for different length tapes

    # Create a temporary directory to store multiple .npy files
    temp_dir = tempfile.mkdtemp()
    tape_files = []

    # Create multiple trajectory tapes
    for i in range(n_tapes):
        data = np.random.rand(time_steps, n_particles, state_dim)
        tape_path = os.path.join(temp_dir, f"tape_{i}.npy")
        np.save(tape_path, data)
        tape_files.append(tape_path)

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp_seq:
        # Create a sample amino acid sequence
        sequence = "ACD"  # One letter for each particle
        tmp_seq.write(sequence)

    yield {
        "data_path": temp_dir,  # Now pointing to directory containing tapes
        "particle_identities_path": tmp_seq.name,
        "expected_shapes": {
            "n_tapes": n_tapes,
            "tape_length": tape_length,
            "n_particles": n_particles,
            "state_dim": state_dim,
            "n_windows": n_windows,
            "window_size": window_size,
        },
    }

    # Cleanup temporary files
    for file in tape_files:
        os.unlink(file)
    os.rmdir(temp_dir)
    os.unlink(tmp_seq.name)


def test_dataset_initialization(sample_data):
    dataset = TimeSeriesDataset(
        sample_data["data_path"],
        sample_data["particle_identities_path"],
        window_size=sample_data["expected_shapes"]["window_size"],
    )

    assert len(dataset) == sample_data["expected_shapes"]["n_windows"]
    assert isinstance(dataset.particle_labels, torch.Tensor)
    assert len(dataset.particle_labels) == sample_data["expected_shapes"]["n_particles"]


def test_dataset_getitem(sample_data):
    dataset = TimeSeriesDataset(
        sample_data["data_path"],
        sample_data["particle_identities_path"],
        window_size=sample_data["expected_shapes"]["window_size"],
    )

    initial_condition, targets, particle_labels = dataset[0]

    # Check shapes
    assert initial_condition.shape == (
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert targets.shape == (
        sample_data["expected_shapes"]["window_size"] - 1,
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert particle_labels.shape == (sample_data["expected_shapes"]["n_particles"],)

    # Check types
    assert isinstance(initial_condition, torch.FloatTensor)
    assert isinstance(targets, torch.FloatTensor)
    assert isinstance(particle_labels, torch.Tensor)


def test_dataloader(sample_data):
    batch_size = 4
    window_size = sample_data["expected_shapes"]["window_size"]
    dataloader = create_dataloader(
        sample_data["data_path"],
        sample_data["particle_identities_path"],
        batch_size,
        window_size=window_size,
    )

    # Get first batch
    batch = next(iter(dataloader))
    initial_condition, targets, particle_labels = batch

    # Check batch shapes
    assert initial_condition.shape == (
        batch_size,
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert targets.shape == (
        batch_size,
        window_size - 1,  # targets are window_size - 1 timesteps
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert particle_labels.shape == (
        batch_size,
        sample_data["expected_shapes"]["n_particles"],
    )
