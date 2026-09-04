# slurm/

Job files used for the reported runs (SuperMIC-style Slurm). Before use, set
`#SBATCH -A`, and export `ASI_ROOT` (directory with `data/` = raw trace, `agg/` =
aggregates) and `REPO` (this repository). `PY` defaults to `python`; it must have the
packages in `requirements.txt`. Login-node time limits apply on most clusters: run
everything through `sbatch`.

* `a1agg_job.sh` — stage 0 aggregation (raw scan, ~4 h on 20 cores)
* `a3_job.sh`, `a3v2_job.sh`, `a3audit_job.sh` — span extraction, scheduler analysis, join audit
* `a2_job.sh`, `a5_job.sh`, `a6_job.sh`, `a8_job.sh` — single stages
* `rerun_p0_job.sh` — chain 1 → 5 → 6 → 8 plus the two re-anchored sensitivities
* `prewrite_job.sh` — stage 9, stage 8 tiers, stage 3c
* `a7_sensitivity_job.sh` — stages 1+5 with the re-anchored configs
