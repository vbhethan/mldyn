"""Functions to coarse-grain protein structures
for use in the downstream dynamics learning pipeline"""

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm


def generate_ca_cb_indices(u: mda.Universe):
    """
    Given input universe object, returns the paired indices of the CA and CB atoms
    in the protein in the order of the residues
    """
    ca_indices = []
    cb_indices = []
    protein_selection = u.select_atoms("protein")

    for residue in protein_selection.residues:
        if residue.resname == "GLY":
            ca_indices.append(residue.atoms.select_atoms("name CA").indices[0])
            cb_indices.append(residue.atoms.select_atoms("name HA1").indices[0])
        else:
            ca_indices.append(residue.atoms.select_atoms("name CA").indices[0])
            cb_indices.append(residue.atoms.select_atoms("name CB").indices[0])
    return ca_indices, cb_indices


def generate_coarse_grained_structure(timestep, ca_indices, cb_indices):
    """
    Takes a timestep positions and the associated indices of the CA and CB atoms
    returns the coarse grained representation as defined by the position of the
    CA atom and the vector pointing from the CA to the CB atom
    """

    coarse_grained_positions = np.concatenate(
        (
            timestep.positions[ca_indices],
            timestep.positions[cb_indices] - timestep.positions[ca_indices],
        ),
        axis=1,
    )

    return coarse_grained_positions


def gen_cg_trajectory(u: mda.Universe):
    """
    Given a universe object, returns the coarse grained trajectory
    Output shape should be (n_frames, n_residues, 6)
    """
    ca_indices, cb_indices = generate_ca_cb_indices(u)
    cg_traj = []
    for timestep in tqdm(u.trajectory):
        cg_traj.append(
            generate_coarse_grained_structure(timestep, ca_indices, cb_indices)
        )
    return np.array(cg_traj)
