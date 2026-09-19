#!/bin/bash
#SBATCH -A hpc_gridopt02
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -x smic002
#SBATCH -J flexdecomp
#SBATCH -o /work/mli30/res-gpu-power/2026-09-15_flexibility-decomposition/slurm-%j.out
set -euo pipefail
module load python/3.11.5-anaconda 2>/dev/null || true
# absolute path: smic002 has a broken modules install (jobs 571986/571990)
PY=/usr/local/packages/python/3.11.5-anaconda/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CODE=/work/mli30/res-gpu-power
OUT=/work/mli30/res-gpu-power/2026-09-15_flexibility-decomposition
cd "$CODE"
mkdir -p "$OUT"

$PY src/flex_decomposition.py \
  --agg /project/mli30/mli30/asi-trace/agg/pod_hourly_agg.parquet \
  --config configs/power_curves.yaml \
  --out "$OUT" \
  --nboot 1000 --n-ensemble 200 \
  --n-shift 12 --n-shift-boot 3 \
  --helios-data /work/mli30/res-gpu-power/external-data/helios/HeliosData/data

echo FLEXDECOMP_DONE
