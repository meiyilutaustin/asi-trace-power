#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 12:00:00
#SBATCH -J a1agg
#SBATCH -o a1agg-%j.out

PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export OUT_DIR=$BASE/a1_out
export POWER_CFG=$REPO/configs/power_curves.yaml
export NPROC=18

set -e
$PY $REPO/src/aggregate_pod_hourly.py
$PY $REPO/src/a1_load_reconstruction.py
echo "A1AGG_PIPELINE_DONE"
