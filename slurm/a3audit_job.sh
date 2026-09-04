#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 04:00:00
#SBATCH -J a3audit
#SBATCH -o a3audit-%j.out
PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export OUT_DIR=$BASE/a3audit_v2
export POWER_CFG=$REPO/configs/power_curves.yaml
set -e
$PY $REPO/src/a3_join_audit.py
echo "A3AUDIT_DONE"
