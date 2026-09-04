#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 12:00:00
#SBATCH -J a3battery
#SBATCH -o a3battery-%j.out

PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export OUT_DIR=$BASE/a3_out
export POWER_CFG=$REPO/configs/power_curves.yaml
export NPROC=10

set -e
$PY $REPO/src/a3_extract_pod_spans.py
$PY $REPO/src/a3_battery_inversion.py
echo "A3_PIPELINE_DONE"
