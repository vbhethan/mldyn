PYTHON_EXE=$(which python)
DATAFILE="./sim_data/example_data.npy"
DATAFILE_BASENAME="example_data"

$PYTHON_EXE train.py \
    --no-cuda \
    --epochs 10 \
    --num-atoms 34 \
    --dims 3 \
    --save-folder logs \
    --skip-first \
    --datafile-basename $DATAFILE_BASENAME \
    --data-dir "./sim_data"
