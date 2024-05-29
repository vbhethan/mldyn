#!/bin/bash -l
#SBATCH --job-name=qm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=5000
#SBATCH -t 1-00:00:00
#SBATCH -p sunshine
#SBATCH -A WAG
#SBATCH --gres=gpu:1

mkdir -p logs

$PYTHON_EXE train.py \
--epochs 150 \
--num-atoms 148 \
--dims 6 \
--save-folder logs \
--skip-first \
--datafile-basename "3cln_combined_cg_trj" \
--data-dir "./sim_data"
