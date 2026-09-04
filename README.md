# asi-trace-power

Code, configuration, derived data products and result tables for

> **Beyond Scalar Flexibility: From Eligible AI Workloads to Dependable Load Relief**
> (Meiyi Li, Louisiana State University; ACM e-Energy 2027 submission; preprint link to be added).

The pipeline turns the public Alibaba `cluster-trace-gpu-v2026` (per-pod hourly GPU
utilisation, job type, priority, scheduling delay; 185 days) into an hourly electric
load profile with Monte Carlo uncertainty, a workload-semantic flexibility envelope, and
a duration–reliability–portfolio surface of *eligible* curtailment, and then evaluates
that surface against the fixed-percentage representations used in grid studies.

Everything reported in the paper is reproducible from this repository plus two public
inputs that we do not redistribute (see *Data*).

## Layout

```
src/            analysis pipeline (Python 3.11; pandas, pyarrow, numpy, matplotlib, pyyaml)
configs/        util->power model parameters (power_curves.yaml) and two re-anchored variants
slurm/          job templates used for the reported runs (edit ACCOUNT / ASI_ROOT / REPO)
data/external/  EIA-930 sub-region 8910 (MISO South) hourly demand 2024-2025 (public domain)
data/validation/ compiled published GPU power measurements (117 points) and daily-shape metrics
data/products/  derived hourly products (fleet band, envelopes, K surface; see data/README.md)
results/        per-stage summary.json, tables and figures as used in the paper
figures/        paper-ready versions of the main figures
```

## Pipeline

| Stage | Script | Reads | Produces | Paper |
|---|---|---|---|---|
| 0 | `aggregate_pod_hourly.py` | raw `pod_hourly` (351 GB) + `server_hourly` | `pod_hourly_agg.parquet` (858,816 rows, sufficient statistics S_b) | Methods 4.2 |
| 1 | `a1_load_reconstruction.py` | stage 0 + `configs/power_curves.yaml` | hourly facility band (P5/P50/P95), components | Fig. 1, Table 2 |
| 2 | `a2_fluctuation_scaling.py` | stage 0 | Taylor exponents, synchrony ratios (raw / deseasonalised / capacity-normalised) | Results §2 |
| 3a | `a3_extract_pod_spans.py` | raw `pod_hourly` | per-pod execution spans (77.8 M) | Methods 4.6 |
| 3b | `a3_join_audit.py`, `a3_matched_unmatched_audit.py` | 3a + `job_execution_summary` | join coverage by day/label, change-point of the difference process | Methods 4.6, App. |
| 3c | `a3_battery_inversion.py` | 3a + summary | run-on-arrival counterfactual, backlog replay of observed delay | App. (backlog) |
| 5 | `a5_envelope.py` | stage 0 + summary | four-layer envelope on three electrical boundaries, three eligibility mappings, MC bands | Results §3, Fig. 3 |
| 6 | `a6_accreditation.py` | stage 0 + summary | K(α,h,S) surface, portfolios, paired counterfactuals, eras, holdout | Results §4, Fig. 4, Tables 3–4 |
| 6′ | `a6_tail_scalar.py`, `a6_required_share.py` | stage 6 / 5 outputs | tail-calibrated scalar, required-share grid | Table 4, App. |
| 7 | `a7_validation_plot.py` | `data/validation/` | model anchors vs measurements, daily-amplitude comparison | Methods 4.3, App. |
| 8 | `a8_headroom.py` | stages 1, 5 + `data/external/` | illustrative peak-cap screening (MISO South) | App. |
| 9 | `a9_formulations.py` | stages 1, 5, 6 | published formulations on a common boundary, repeat-event budget, host add-on | Table 5, App. |
| 10 | `a10_jensen_and_online_curve.py` | config + validation | Jensen bound of the piecewise-linear model, online-inference curve | Methods, App. |
| 11 | `a11_intrahour_minima.py` | NLR raw NVML logs (not redistributed) | within-hour minima relative to the hourly mean | App. |
| r1 | `r1_realizable_sweep.py` | stages 1, 5 + external | realizable fraction q, rebound energy, recovery, screening vs q | Results §5, App. |
| r2 | `r2_rolling_origin.py` | stage 5 | rolling-origin validation of K, derating factors | Results §4, App. |
| r4 | `r4_covariance_decomposition.py` | stage 5 (cluster series) | exact covariance decomposition, capacity plateaus, paired bootstrap | Results §2, App. |
| — | `replot_v2.py` | stage 5/6 outputs | paper-ready figures in `figures/` | Figs. 3–4 |

Stages 0 and 3a scan the raw trace and need a cluster (they took ~4 h and ~2 h on 20
cores); every other stage reads the 49 MB aggregate table or smaller products and runs
in minutes on a laptop. `slurm/` holds the job files we used; set `ASI_ROOT` (directory
containing `data/` = raw trace and `agg/` = aggregates) and `REPO`.

## Reproducing the paper numbers

1. Download the four archives of `cluster-trace-gpu-v2026` from the Alibaba ClusterData
   repository into `$ASI_ROOT/data/` and unzip (~332 GB). We do not redistribute the trace.
2. `python src/aggregate_pod_hourly.py` → `$ASI_ROOT/agg/pod_hourly_agg.parquet`
   (or download `pod_hourly_agg.parquet` from the v1.0.0 GitHub release and skip 1–2).
3. Run stages 1, 5, 6, 8, 9 in that order with the base config, then r1/r2/r4 and the
   6′ post-processing; stage 3 needs the span extraction first. Each script writes
   `summary.json` plus figures to `$OUT_DIR`.
4. `results/headline.json` collects every number quoted in the paper with its source
   stage; `results/<stage>/summary.json` are the stage outputs it was built from.

Model conventions that matter for interpretation are documented in
`src/powermodel.py` (idle fraction φ, online-inference floor c0, the three curtailment
boundaries *attributed / idle_retained / node_sleep*, the three eligibility mappings)
and in `configs/power_curves.yaml`.

## Data

* **Raw trace**: Alibaba `cluster-trace-gpu-v2026`, released with Li et al., *Heterogeneity
  at Hyperscale*, OSDI 2026. Research/study use per the ClusterData repository; obtain it
  from the authors' object storage. Not redistributed here.
* **Regional demand**: U.S. EIA Hourly Electric Grid Monitor (EIA-930), sub-region 8910,
  2024–2025, public domain; a filtered copy is in `data/external/`.
* **Published GPU power measurements** used for external validation are compiled in
  `data/validation/validation_points.csv` with source, page/table and exactness tag; the
  NREL/NLR raw NVML dataset (DOI 10.7799/3025227) is not redistributed.
* **Derived products** (`data/products/`): hourly fleet power band, fleet and per-cluster
  envelopes, the K surface, and the server aggregate. The 49 MB pod aggregate table with
  sufficient statistics is attached to the `v1.0.0` GitHub release rather than tracked in
  git; an archival (Zenodo) DOI for the release will be added on acceptance.

Derived products inherit the research/study-use condition of the source trace; please cite
both the OSDI 2026 trace paper and this work.

## License

Code: MIT (see `LICENSE`). Derived data and result tables: CC BY-NC 4.0, research and study
use, subject to the source-trace terms (`DATA_LICENSE.md`).

## Citation

See `CITATION.cff`. Preprint and archive DOIs will be added when available.
