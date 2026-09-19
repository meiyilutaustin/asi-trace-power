# asi-trace-power

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22308422.svg)](https://doi.org/10.5281/zenodo.22308422)

*Concept DOI [10.5281/zenodo.22308422](https://doi.org/10.5281/zenodo.22308422) always resolves to the latest version. This revision is **v1.1.0**, version DOI [10.5281/zenodo.22839997](https://doi.org/10.5281/zenodo.22839997).*

Code, configuration, derived data products and result tables for

> **Beyond Scalar Flexibility: From Eligible AI Workloads to Dependable Load Relief**
> (Meiyi Li, Louisiana State University; ACM e-Energy 2027 submission; preprint link to be added).

> **This revision.** This is the revised paper artifact (e-Energy 2027, the
> *scope → persistence → coincidence* decomposition), released as **v1.1.0**
> (internal revision `paper-D-r2-2026-09-18`), archived at Zenodo, version DOI
> [10.5281/zenodo.22839997](https://doi.org/10.5281/zenodo.22839997). The previous
> release was v1.0.1 (DOI 10.5281/zenodo.22308423).

The pipeline turns the public Alibaba `cluster-trace-gpu-v2026` (per-pod hourly GPU
utilisation, job type, priority, scheduling delay; 185 days) into an hourly electric load
profile with Monte Carlo uncertainty, and then asks how much of that load is *dependable*
demand flexibility. The central object is a **decomposition of flexibility that a single
scalar (e.g. "20% of load is curtailable") hides**:

1. **Scope** — what fraction of the workload is *eligible* to be curtailed at all
   (workload-semantic envelope: floor / shift / curtail / standby layers).
2. **Persistence** — how much of the eligible power actually survives a required
   *duration* (1 h / 4 h / 24 h).
3. **Coincidence** — how much a *portfolio* of clusters loses because troughs do not line
   up in time (observed vs an independent-timing benchmark).

Each step multiplies the last (an exact identity, `flex_decomposition`), and *duration* is
the common driver of the persistence and coincidence penalties. The paper then evaluates
this dependable-relief surface against the fixed-percentage representations used in grid
studies, on a common electrical boundary, with a blind cross-month validation and a
regional peak-cap screen.

Everything reported in the paper is reproducible from this repository plus two public
inputs that we do not redistribute (see *Data*).

## Layout

```
src/            analysis pipeline + revised-scope scripts (Python 3.11; pandas, pyarrow, numpy, matplotlib, pyyaml)
configs/        util->power model parameters (power_curves.yaml) and two re-anchored variants
slurm/          job templates for the runs that scan the raw trace (edit ACCOUNT / ASI_ROOT / REPO)
data/external/  EIA-930 sub-region 8910 (MISO South) hourly demand 2024-2025 (public domain)
data/validation/ compiled published GPU power measurements and daily-shape metrics
data/products/  derived hourly products (fleet band, envelopes, K surface; see data/README.md)
results/        per-stage outputs and headline.json as used in the paper (see results/README.md)
figures/        paper-ready Python-base figures (fig1..fig7); the hand-annotated composites live in the paper only
```

## Pipeline

Stages are grouped by the three decomposition steps. `results/README.md` maps every stage
directory to its script and the canonical run it was produced from.

| Stage | Script | Produces | Paper |
|---|---|---|---|
| 0 | `aggregate_pod_hourly.py` | `pod_hourly_agg.parquet` (sufficient statistics S_b) | Methods |
| 1 — load | `a1_load_reconstruction.py` | hourly facility band (P5/P50/P95), growth, components | Fig. 1, Fig. 2 |
| 2 — synchrony | `a2_fluctuation_scaling.py`, `covariance_robustness_v2.py`, `r4_covariance_decomposition.py` | Taylor exponents, exact covariance decomposition, matched portfolios | Results §synchrony |
| 3 — join audit | `a3_extract_pod_spans.py`, `a3_join_audit.py`, `a3_matched_unmatched_audit.py` | execution-join coverage (data-limitation caveat only) | Methods, App. |
| Scope | `a5_envelope.py`, `shift_conditional.py` | four-layer eligible-workload envelope, rebound/recovery | Fig. 2, Fig. 4 |
| Persistence + coincidence | `flex_decomposition.py`, `flex_decomposition_deliverables.py` | the scope→persistence→coincidence identity (E0–E4) | Fig. 3, Fig. 6 |
| Availability surface | `a6_accreditation.py`, `covariance_robustness_v2.py` | K(α,h,S) surface, portfolio firmness, diversification | Fig. 4, Tables |
| Coincidence controls | `coincidence_controls.py`, `reviewB_b2_pind_robustness.py` | negative controls (truncation / single-cluster / phase-random) | Fig. 5, App. |
| Portfolio scan | `fig7a_portfolio_scan.py` | 4 h availability vs clusters pooled (Fig. 7a) | Fig. 7 |
| Tail events | `tail_event_dedup.py` | de-duplicated trough events, forward-split audit, derating trade-off | App. |
| Validation | `a7_validation_plot.py`, `m100_validation_v2.py` | anchors vs measurements; **blind cross-month M100** | Methods, App. |
| Peak-cap screen | `a8_headroom.py` | MISO South interconnection screen at a 25% cluster-power cap | Fig. 7, App. |
| Backtests | `r1_realizable_sweep.py`, `r2_rolling_origin.py` | realizable fraction q; rolling-origin derating | Results, App. |
| Review bundles | `reviewB_*.py`, `reviewC_*.py` | independent-calibration backtest, M100 bin propagation, facility-scope, cluster-set consistency | App. |
| Figures | `make_main_figures_prep.py`, `make_main_figures.py`, `fig2_envelope_dualband.py`, `fig7_make.py`, `replot_v2.py` | the Python-base figures in `figures/` | Figs. 1–7 |
| — | `build_headline.py` | `results/headline.json` from the stage files | — |

Stages 0 and 3 (span extraction) scan the raw trace and need a cluster (~4 h and ~2 h on
20 cores); `slurm/` holds the job files used. Every other stage reads the 49 MB aggregate
table or the derived products in `data/products/` and runs in minutes on a laptop.

## Figures

`figures/` ships the **Python-base** (un-annotated) figures, regenerable from the
scripts above:

| File | Figure | Script |
|---|---|---|
| `fig1_concept.{pdf,png}` | Fig. 1 — concept | author schematic (base in `make_main_figures.py`) |
| `fig2_envelope_dualband.{pdf,png}` | Fig. 2 — dual-band envelope | `fig2_envelope_dualband.py` |
| `fig3_decomposition.{pdf,png}` | Fig. 3 — scope→persistence→coincidence funnels | `make_main_figures.py` |
| `fig4_scope.{pdf,png}` | Fig. 4 — eligible scope | `make_main_figures.py` |
| `fig5_coincidence.{pdf,png}` | Fig. 5 — coincidence | `make_main_figures.py` |
| `fig6_driver.{pdf,png}` | Fig. 6 — duration driver | `make_main_figures.py` |
| `fig7_decisions.{pdf,png}` | Fig. 7 — three decisions (portfolio scan panel a) | `fig7_make.py`, `fig7a_portfolio_scan.py` |

The figures printed in the paper are hand-annotated composites of these bases (titles and
callouts added in a slide editor); the annotation step is not part of this repository.

## Reproducing the paper numbers

1. Download the four archives of `cluster-trace-gpu-v2026` from the Alibaba ClusterData
   repository into `$ASI_ROOT/data/` and unzip (~332 GB). We do not redistribute the trace.
2. `python src/aggregate_pod_hourly.py` → `$ASI_ROOT/agg/pod_hourly_agg.parquet`
   (or download `pod_hourly_agg.parquet` from the GitHub release and skip 1–2).
3. Run the load and scope stages (1, `a5_envelope`, `shift_conditional`), then the
   availability surface (`a6_accreditation`, `covariance_robustness_v2`) and the
   decomposition (`flex_decomposition`); then the controls, tail, portfolio scan, peak-cap
   screen, validation and backtests. Each script writes its `summary.json` to `$OUT_DIR`.
4. `python src/build_headline.py` collects every quoted number into `results/headline.json`
   with its source stage; `results/<stage>/` are the stage outputs it is built from.

Model conventions that matter for interpretation are documented in `src/powermodel.py`
(idle fraction φ, online-inference floor c0, the three curtailment boundaries
*attributed / idle_retained / node_sleep*, the three eligibility mappings) and in
`configs/power_curves.yaml`. Three power denominators — workload / IT / facility — are kept
distinct throughout; all MW are GPU-side workload power unless a boundary is named.

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
  sufficient statistics is attached to the GitHub releases rather than tracked in git;
  releases are archived at Zenodo under concept DOI 10.5281/zenodo.22308422 (v1.1.0 =
  10.5281/zenodo.22839997).

Derived products inherit the research/study-use condition of the source trace; please cite
both the OSDI 2026 trace paper and this work.

## License

Code: MIT (see `LICENSE`). Derived data and result tables: CC BY-NC 4.0, research and study
use, subject to the source-trace terms (`DATA_LICENSE.md`).

## Citation

See `CITATION.cff`. Archived release: https://doi.org/10.5281/zenodo.22839997 (v1.1.0); concept DOI (all versions) https://doi.org/10.5281/zenodo.22308422. Preprint link to be added.
