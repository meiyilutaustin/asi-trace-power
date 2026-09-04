#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH -J rerunP0
#SBATCH -o rerunP0-%j.out
# revision 2 P0 chain: A1 v2 -> A5 v2 -> A6 v2 -> A8 v2 with the base config
# (online floor, idle_retained boundary, central eligibility), then the two A7b
# re-anchored variants (A1+A5 only). Each stage writes to its own *_v2 dir so the
# v1 outputs stay for reconciliation.
PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
export POWER_CFG=$REPO/configs/power_curves.yaml
set -e
export OUT_DIR=$BASE/a1_v2;  $PY $REPO/src/a1_load_reconstruction.py; echo "STAGE_DONE a1"
export OUT_DIR=$BASE/a5_v2;  $PY $REPO/src/a5_envelope.py;            echo "STAGE_DONE a5"
export OUT_DIR=$BASE/a6_v2;  $PY $REPO/src/a6_accreditation.py;       echo "STAGE_DONE a6"
export A1_DIR=$BASE/a1_v2 A5_DIR=$BASE/a5_v2 OUT_DIR=$BASE/a8_v2
$PY $REPO/src/a8_headroom.py; echo "STAGE_DONE a8"
for V in a7_measured_mid a7_measured_low; do
  export POWER_CFG=$REPO/src/power_curves_$V.yaml
  export OUT_DIR=$BASE/a7_v2/$V/a1; $PY $REPO/src/a1_load_reconstruction.py
  export OUT_DIR=$BASE/a7_v2/$V/a5; $PY $REPO/src/a5_envelope.py --mc 60
  echo "STAGE_DONE a7b_$V"
done
echo "RERUN_P0_DONE"
