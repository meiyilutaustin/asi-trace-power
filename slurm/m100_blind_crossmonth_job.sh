#!/bin/bash
#SBATCH -A hpc_gridopt02
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 08:00:00
#SBATCH -x smic002
#SBATCH -J m100blind
#SBATCH -o /work/mli30/res-gpu-power/2026-09-15_m100-blind-crossmonth/slurm-%j.out
set -euo pipefail
module load python/3.11.5-anaconda 2>/dev/null || true
# absolute path: smic002 has a broken modules install (jobs 571986/571990)
PY=/usr/local/packages/python/3.11.5-anaconda/bin/python
# DuckDB (and any pinned deps) live in the repo toolchain, not the base env.
export PYTHONPATH=/project/mli30/mli30/_toolchains/res-gpu-power/external-pydeps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CODE=/work/mli30/res-gpu-power
DATA=/work/mli30/res-gpu-power/m100/raw
OUT=/work/mli30/res-gpu-power/2026-09-15_m100-blind-crossmonth
cd "$CODE"
mkdir -p "$OUT/cache"

# Comprehensive cross-month design (all months now available): calibrate on
# March 2022; evaluate July (already-inspected, for continuity) AND the held-out
# September 2022 panel (never inspected = the blind cross-month test). Room-PUE
# facility comparison on Dec 2021 / March / September.
$PY src/m100_validation_v2.py \
  --data "$DATA" \
  --cache "$OUT/cache" \
  --out "$OUT" \
  --months 22-03 22-07 22-09 \
  --facility-months 21-12 22-03 22-09 \
  --threads 16

echo M100_BLIND_DONE
