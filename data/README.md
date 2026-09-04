# data/

| path | content | source |
|---|---|---|
| `external/eia930_miso_south_8910_hourly_2024_2025.csv` | hourly demand, EIA-930 sub-region 8910 (MISO South), 2024–2025; 47 missing hours are interpolated by the scripts | U.S. EIA Hourly Electric Grid Monitor six-month CSVs (public domain) |
| `validation/validation_points.csv` | 117 published GPU/node power measurements (fraction of TDP) with source, GPU, workload phase and an EXACT / COMPUTED / APPROX tag | see `source` column |
| `validation/diurnal_profiles.csv` | daily-shape metrics from public traces, papers and simulations | see `source` column |
| `validation/nlr_*.csv` | our re-analysis of the NREL/NLR raw NVML dataset (DOI 10.7799/3025227) | computed; raw logs not redistributed |
| `products/fleet_hourly_power.parquet` | hourly facility power: MC P5/P50/P95, base-parameter value, IT P50; `have` flags valid hours | stage 1 |
| `products/envelope_hourly.parquet` | fleet four-layer envelope per hour, curtailment on three boundaries, horizon-specific shift layers | stage 5 |
| `products/envelope_cluster_hourly.parquet` | the same per logical cluster (long format) | stage 5 |
| `products/k_surface.parquet` | K(α,h) for fleet, enumerated portfolios, eras; observed and counterfactual columns | stage 6 |
| `products/server_hourly_agg.parquet` | server inventory aggregate (GPU count, CPU cores) per hour × cluster × GPU model | stage 0 |

The 858,816-row pod aggregate table (`pod_hourly_agg.parquet`, 49 MB) that every
downstream stage reads is provided in the archived data release (DOI to be added); place it
under `$ASI_ROOT/agg/`. All MW in the products are GPU-side workload power unless the column
name states a boundary; see `src/powermodel.py`.
