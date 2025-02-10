import pytest
import numpy as np
import torch
import tempfile
import os
from mldyn.data.dataloaders import TimeSeriesDataset, create_dataloader


@pytest.fixture
def sample_data():
    # Create temporary test data
    n_windows = 10
    time_steps = 5
    n_particles = 3
    state_dim = 6  # e.g., x, y, z coordinates + r sidechain

    data = np.random.rand(n_windows, time_steps, n_particles, state_dim)

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_data:
        np.save(tmp_data, data)

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp_seq:
        # Create a sample amino acid sequence
        sequence = "ACD"  # One letter for each particle
        tmp_seq.write(sequence)

    yield {
        "data_path": tmp_data.name,
        "particle_identities_path": tmp_seq.name,
        "expected_shapes": {
            "n_windows": n_windows,
            "time_steps": time_steps,
            "n_particles": n_particles,
            "state_dim": state_dim,
        },
    }

    # Cleanup temporary files
    os.unlink(tmp_data.name)
    os.unlink(tmp_seq.name)


def test_dataset_initialization(sample_data):
    dataset = TimeSeriesDataset(
        sample_data["data_path"], sample_data["particle_identities_path"]
    )

    assert len(dataset) == sample_data["expected_shapes"]["n_windows"]
    assert isinstance(dataset.particle_labels, torch.Tensor)
    assert len(dataset.particle_labels) == sample_data["expected_shapes"]["n_particles"]


def test_dataset_getitem(sample_data):
    dataset = TimeSeriesDataset(
        sample_data["data_path"], sample_data["particle_identities_path"]
    )

    initial_condition, targets, particle_labels = dataset[0]

    # Check shapes
    assert initial_condition.shape == (
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert targets.shape == (
        sample_data["expected_shapes"]["time_steps"] - 1,
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
    dataloader = create_dataloader(
        sample_data["data_path"], sample_data["particle_identities_path"], batch_size
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
        sample_data["expected_shapes"]["time_steps"] - 1,
        sample_data["expected_shapes"]["n_particles"],
        sample_data["expected_shapes"]["state_dim"],
    )
    assert particle_labels.shape == (
        batch_size,
        sample_data["expected_shapes"]["n_particles"],
    )
