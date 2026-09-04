#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -J a5envelope
#SBATCH -o a5envelope-%j.out

PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export OUT_DIR=$BASE/a5_out
export POWER_CFG=$REPO/configs/power_curves.yaml

set -e
$PY $REPO/src/a5_envelope.py
echo "A5_PIPELINE_DONE"
