PYTHON_EXE="/Users/vbhethan/miniconda3/envs/torch/bin/python"
mkdir -p logs

$PYTHON_EXE train.py \
--no-cuda \
--epochs 10 \
--num-atoms 34 \
--dims 3 \
--save-folder logs \
--skip-first \
--datafile-basename "example_data" \
--data-dir "./sim_data"
