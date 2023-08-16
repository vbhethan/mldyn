import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
import numpy as np



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

    examples = np.zeros((n_examples, num_frames_per_example, trajectory.shape[1], trajectory.shape[2]))

    for i in range(n_examples):
        examples[i] = trajectory[i * num_frames_per_example : (i + 1) * num_frames_per_example]

    return examples