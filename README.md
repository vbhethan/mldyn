# Autoregressive transformer for multi-step dynamics forecasting with latent edges
Vignesh C. Bhethanabotla

## Overview
This repo trains an autoregressive transformer on molecular dynamics trajectories of polypeptides/proteins to forecast future states and reveal interaction pathways. By analyzing the model’s attention/latent edges through a Granger-causality lens, we probe how residue interactions propagate and influence downstream motion. It includes preprocessing utilities (with the MDAnalysis package) to coarse-grain trajectories and prepare inputs (works with standard protein residue/atom naming; tested on GROMACS outputs), plus postprocessing tools to analyze trained models and generate visualizations.

## Installation
The package can be installed using pip with the following options
- Base (CPU PyTorch): `pip install .` or `pip install ".[cpu]"`
- For CUDA support (PyTorch 2.2.2 cu121):  
  `pip install --extra-index-url https://download.pytorch.org/whl/cu121 ".[cuda]"`
- To include Preprocessing tools (MDAnalysis, h5py, scipy, tqdm): `pip install ".[preprocess]"`
- Postprocessing/visualization (matplotlib, tqdm): `pip install ".[postprocess]"`
- Dev tools (pytest, black, ruff): `pip install ".[dev]"`

Extras can be combined, e.g. `pip install ".[cpu,preprocess,postprocess]"`.

For example, to install with CUDA support and with pre- and post-processing utilities, run the following after navigating to the project directory

`pip install --extra-index-url https://download.pytorch.org/whl/cu121 ".[cuda,preprocess,postprocess]"`

## Quick start / Example run
Example data (short polypeptide MD trajectories) are provided in `sim_data`. To train on the example set:
`python train.py`
Hyperparameters and data layout are defined in `train.py`.
After training, run the postprocessing demo:
`python post-process.py`

## Data prep
You can use the functions in the `preprocessing` module to prepare input data from MD trajectories. the `gen_cg_trajectory` function takes an `MDAnalysis.Universe` and generates a `numpy.ndarray` containing the coarse-grained dynamics. You will also need to provide a file specifying the residue sequence of the polypeptide using the 1-letter amino-acid codes (similar to the sequence specification in the FASTA format, see the `particle_identities.txt` example file provided in `sim_data`)

Example: coarse-grain an MDAnalysis trajectory and save to `.npy`:

```python
import numpy as np
import MDAnalysis as mda
from mldyn.preprocessing.cgutils import gen_cg_trajectory

# Replace with your topology/trajectory paths (any format supported by MDAnalysis should work)
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Generate coarse-grained positions: shape (n_frames, n_residues, 6)
cg = gen_cg_trajectory(u)

# Save one tape; create one file per independent run
np.save("sim_data/run1.npy", cg)

# Create particle_identities.txt once with the residue sequence aligned with the coordinates from the trajectory
# for example, using MDAnalysis to produce the string of amino-acid letter codes
protein = u.select_atoms("protein")
seq = protein.residues.sequence(format="string")
with open("sim_data/particle_identities.txt", "w") as f:
    f.write(seq)
```

## Training
Refer to `train.py` in the base directory for an example of how to set up a training run.
- Minimal inputs: coarse-grained `.npy` tapes (one per run) in a data directory (e.g., `sim_data/`) and `particle_identities.txt` matching the residue order.
- Make sure to set the hyperparameters as in `train.py`, which are currently set to match the example data provided; set them to match your data and specifications:

```python
# train.py 
n_particles = 20              # number of residues
input_state_dimension = 6     # CA coords + CA->CB vector
d_model = 128
n_particle_types = 20
n_time_steps = 19             # window_size - 1 targets
d_feedforward = 256
num_epochs = 100
learning_rate = 1e-4
batch_size = 4

# Paths to datafiles
train_data_path = "./sim_data/"
particle_identities_path = "./sim_data/particle_identities.txt"
```

- Run training: `python train.py` (will use CUDA if available); saves `model.pth`.

## Postprocesing    
Refer to `post-process.py` for an end-to-end example. It loads a trained model with attention outputs enabled, runs over the dataset, aggregates encoder/self/cross attention across samples, and saves the averaged weights to `combined_attention_weights.npy`. Running:
`python post-process.py`
will load `model.pth`, read data from `./sim_data/`, and write the aggregated attention maps, which can then be analyzed / visualized to examine the learned interactions between the residues. 

## Repository Structure
- `mldyn/`: core library
  - `models/transformer.py`: autoregressive transformer definition
  - `data/dataloaders.py`: windowed trajectory loader for `.npy` tapes
  - `preprocessing/cgutils.py`: coarse-graining utilities (CA/CB)
  - `postprocessing/transformer_postprocess_*`: attention extraction/aggregation
  - `layers/layers.py`: encoder/decoder building blocks; `loss.py`
- `train.py`: example training script (uses `sim_data/` defaults)
- `post-process.py`: example postprocessing script (averages attention maps)
- `sim_data/`: sample trajectories and `particle_identities.txt`
- `tests/`: basic tests for dataloader and model