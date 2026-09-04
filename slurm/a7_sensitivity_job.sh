#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -J a7sens
#SBATCH -o a7sens-%j.out
# A7b: rerun A1 reconstruction + A5 envelope with power curves re-anchored to
# measured H100 levels (two variants), to quantify how the headline numbers move.
PY=${PY:-python}   # python with pandas, pyarrow, numpy, matplotlib, pyyaml
BASE=${ASI_ROOT:?set ASI_ROOT to the directory holding data/ and agg/}
REPO=${REPO:?set REPO to this repository}
export DATA_DIR=$BASE/data
export AGG_DIR=$BASE/agg
set -e
for V in a7_measured_mid a7_measured_low; do
  export POWER_CFG=$REPO/src/power_curves_$V.yaml
  export OUT_DIR=$BASE/a7_out/$V
  mkdir -p $OUT_DIR
  $PY $REPO/src/a1_load_reconstruction.py
  $PY $REPO/src/a5_envelope.py
  echo "VARIANT_DONE $V"
done
echo "A7SENS_DONE"
