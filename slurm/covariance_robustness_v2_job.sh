#!/bin/bash
#SBATCH --job-name=gpu-cov-v2
#SBATCH --account=hpc_gridopt02
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclude=smic002
#SBATCH --output=/work/mli30/res-gpu-power/2026-09-09_covariance-robustness-v2/slurm-%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p /work/mli30/res-gpu-power/external-data/helios/HeliosData
/usr/local/packages/python/3.11.5-anaconda/bin/python -m zipfile -e /work/mli30/res-gpu-power/external-data/helios/data.zip /work/mli30/res-gpu-power/external-data/helios/HeliosData
cd /home/mli30/external-rerun-20260909
/usr/local/packages/python/3.11.5-anaconda/bin/python src/covariance_robustness_v2.py \
  --helios-data /work/mli30/res-gpu-power/external-data/helios/HeliosData/data \
  --asi-pod /project/mli30/mli30/asi-trace/agg/pod_hourly_agg.parquet \
  --asi-server /project/mli30/mli30/asi-trace/agg/server_hourly_agg.parquet \
  --config configs/power_curves.yaml --nboot 1000 --blocks 168 336 \
  --out /work/mli30/res-gpu-power/2026-09-09_covariance-robustness-v2
