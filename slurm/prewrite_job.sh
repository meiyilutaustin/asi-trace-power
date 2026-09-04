#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 04:00:00
#SBATCH -J prewrite
#SBATCH -o prewrite-%j.out
# supplementary analysis,2 (A3 v2 + backlog replay), 4,5,7 (A9), 10 (A8 marginal-PUE tier)
PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data AGG_DIR=$BASE/agg POWER_CFG=$REPO/configs/power_curves.yaml
set -e
export OUT_DIR=$BASE/a9_out A1_DIR=$BASE/a1_v2 A5_DIR=$BASE/a5_v2 A6_DIR=$BASE/a6_v2
$PY $REPO/src/a9_formulations.py; echo "STAGE_DONE a9"
export OUT_DIR=$BASE/a8_v2; $PY $REPO/src/a8_headroom.py; echo "STAGE_DONE a8"
export OUT_DIR=$BASE/a3_v2; $PY $REPO/src/a3_battery_inversion.py; echo "STAGE_DONE a3"
echo "PREWRITE_DONE"
