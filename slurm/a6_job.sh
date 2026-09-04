#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 03:00:00
#SBATCH -J a6accred
#SBATCH -o a6accred-%j.out

PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export OUT_DIR=$BASE/a6_out
export POWER_CFG=$REPO/configs/power_curves.yaml
export A6_BOOT=${A6_BOOT:-200}

set -e
$PY $REPO/src/a6_accreditation.py
echo "A6_DONE"
