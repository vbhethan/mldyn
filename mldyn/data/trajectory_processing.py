import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
import numpy as np
import h5py
from mldyn.data.data_preprocessing import graph_topology_from_pdb


def align_trajectory(topology_file, trajectory_coordinate_file, output_trajectory_file):
    # TODO: it's possible I don't want to align my trajectories prior to training...
    # Will consider removing this function, at least in this package

    mobile_trajectory = mda.Universe(topology_file, trajectory_coordinate_file)

    reference_structure = mda.Universe(topology_file)

    AlignTraj(
        mobile_trajectory,
        reference_structure,
        select="protein and name CA",
        filename=output_trajectory_file,
    ).run()


def trajectory_to_npy(topology_file, trajectory_coordinate_file, npy_fileout):
    """
    Converts a trajectory, using a topology and trajectory file compatible with
    MDAnalysis, to a numpy array
    """

    # For now, only positions TODO: think about velocities
    u = mda.Universe(topology_file, trajectory_coordinate_file)
    ag = u.select_atoms("name CA")

    n_particles = len(ag)
    n_frames = len(u.trajectory)

    trajectory = np.zeros((n_frames, n_particles, 3))

    for ts in u.trajectory:
        trajectory[ts.frame] = ag.positions

    np.save(npy_fileout, trajectory)


def make_examples_from_large_trajectory(trajectory_npy_file, num_frames_per_example):
    """ """

    trajectory = np.load(trajectory_npy_file)

    n_frames = trajectory.shape[0]
    n_examples = n_frames // num_frames_per_example

    examples = np.zeros(
        (n_examples, num_frames_per_example, trajectory.shape[1], trajectory.shape[2])
    )

    for i in range(n_examples):
        examples[i] = trajectory[
            i * num_frames_per_example : (i + 1) * num_frames_per_example
        ]

    return examples


def append_graph_structure_to_hdf5(trajectory_hdf5_file, pdb_file):
    x, edge_index, edge_attr = graph_topology_from_pdb(pdb_file)

    with h5py.File(trajectory_hdf5_file, "r") as f:
        f.create_group("graph")
        f["graph"].create_dataset("x", data=x)
        f["graph"].create_dataset("edge_index", data=edge_index)
        f["graph"].create_dataset("edge_attr", data=edge_attr)


def process_large_trajectory_hdf5_to_examples(
    trajectory_hdf5_file, num_frames_per_example
):
    """
    Given an hdf5 file with top level datasets called "coordinates" and "velocities", make a series of groups identified by their indices
    with datasets with only the number of frames per example
    """

    with h5py.File(trajectory_hdf5_file, "r") as f:
        coordinates = np.array(f["coordinates"][()])
        velocities = np.array(f["velocities"][()])

    n_frames = coordinates.shape[0]
    n_examples = n_frames // num_frames_per_example

    with h5py.File(trajectory_hdf5_file, "a") as f:
        f.create_group("examples")
        for i in range(n_examples):
            f["examples"].create_group(str(i))
            f["examples"][str(i)].create_dataset(
                "coordinates",
                data=coordinates[
                    i * num_frames_per_example : (i + 1) * num_frames_per_example
                ],
            )
            f["examples"][str(i)].create_dataset(
                "velocities",
                data=velocities[
                    i * num_frames_per_example : (i + 1) * num_frames_per_example
                ],
            )
