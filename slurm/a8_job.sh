#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 01:00:00
#SBATCH -J a8headroom
#SBATCH -o a8headroom-%j.out

PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export AGG_DIR=$BASE/agg
export A1_DIR=$BASE/a1_out
export A5_DIR=$BASE/a5_out
export OUT_DIR=$BASE/a8_out
export POWER_CFG=$REPO/configs/power_curves.yaml

set -e
$PY $REPO/src/a8_headroom.py
echo "A8_DONE"
