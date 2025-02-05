import torch
import pytest

from mldyn.models.transformer import TransformerTimeSeriesModel


@pytest.fixture
def model_params():
    return {
        "n_particles": 10,
        "input_state_dimension": 6,
        "d_model": 64,
        "n_particle_types": 5,
        "n_time_steps": 8,
        "d_feedforward": 128,
    }


@pytest.fixture
def samplemodel(model_params):
    return TransformerTimeSeriesModel(**model_params)


def test_model_initialization(samplemodel, model_params):
    """Test if the model initializes with correct parameters"""
    assert isinstance(samplemodel, TransformerTimeSeriesModel)
    assert samplemodel.n_particles == model_params["n_particles"]
    assert samplemodel.input_state_dimension == model_params["input_state_dimension"]
    assert samplemodel.n_time_steps == model_params["n_time_steps"]


def test_forward_pass_shape(samplemodel, model_params):
    """Test if the forward pass returns correct output shape"""
    batch_size = 4
    initial_condition = torch.randn(
        batch_size, model_params["n_particles"], model_params["input_state_dimension"]
    )
    particle_labels = torch.randint(
        0, model_params["n_particle_types"], (batch_size, model_params["n_particles"])
    )

    output = samplemodel(initial_condition, particle_labels)

    # Check output shape: [batch_size, n_time_steps, n_particles, input_state_dimension]
    expected_shape = (
        batch_size,
        model_params["n_time_steps"],
        model_params["n_particles"],
        model_params["input_state_dimension"],
    )
    assert output.shape == expected_shape


def test_batch_processing(samplemodel, model_params):
    """Test if the model can handle different batch sizes"""
    batch_sizes = [1, 4, 8]

    for batch_size in batch_sizes:
        initial_condition = torch.randn(
            batch_size,
            model_params["n_particles"],
            model_params["input_state_dimension"],
        )
        particle_labels = torch.randint(
            0,
            model_params["n_particle_types"],
            (batch_size, model_params["n_particles"]),
        )

        output = samplemodel(initial_condition, particle_labels)
        assert output.shape[0] == batch_size


def test_device_compatibility(samplemodel, model_params):
    """Test if the model works on both CPU and GPU (if available)"""
    batch_size = 4
    initial_condition = torch.randn(
        batch_size, model_params["n_particles"], model_params["input_state_dimension"]
    )
    particle_labels = torch.randint(
        0, model_params["n_particle_types"], (batch_size, model_params["n_particles"])
    )

    # Test on CPU
    output_cpu = samplemodel(initial_condition, particle_labels)
    assert output_cpu.device.type == "cpu"

    # Test on GPU if available
    if torch.cuda.is_available():
        samplemodel = samplemodel.cuda()
        initial_condition = initial_condition.cuda()
        particle_labels = particle_labels.cuda()
        output_gpu = samplemodel(initial_condition, particle_labels)
        assert output_gpu.device.type == "cuda"
