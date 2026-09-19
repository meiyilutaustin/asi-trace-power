# results/

One directory per analysis stage, each holding the `summary.json` (or the named
`*.json`) written by the script, plus any small tables. `headline.json` gathers the
numbers quoted in the paper, keyed by stage; it is rebuilt from the files below by
`python src/build_headline.py` and restates nothing that is not already in a stage file.
`_provenance` in `headline.json` records the model conventions (idle-retained boundary,
central eligibility) and the canonical run each block came from.

Every directory is a verbatim copy of the **final canonical run** for that quantity.
Superseded and exploratory runs are not shipped. Source runs live in the private
`res-gpu-power` project and are named below for traceability.

| directory | script(s) | source run | note |
|---|---|---|---|
| `a1_load` | `a1_load_reconstruction.py` | `2026-09-04_rerun-p0` | facility band, growth, daily profile, component shares |
| `a2_scaling` | `a2_fluctuation_scaling.py`, `covariance_robustness_v2.py` | `2026-09-09_covariance-robustness-v2` | Taylor/synchrony descriptors on allocated-GPU and modeled-MW series |
| `a3_join_audit` | `a3_join_audit.py`, `a3_matched_unmatched_audit.py` | `2026-09-04_rerun-p0`, `2026-09-04_review-followup` | execution-join coverage by day/label; matched-vs-unmatched difference process. **Data-limitation caveat only** — the "scheduler-as-battery" dispatch reading was rejected and is not a result |
| `a5_envelope` | `a5_envelope.py`, `shift_conditional.py` | `2026-09-09_shift-conditional` | eligible scope, shift-conditional recovery, rebound energy and recovery scenarios |
| `a6_availability` | `a6_accreditation.py`, `covariance_robustness_v2.py` | `2026-09-04_rerun-p0` + `2026-09-09_covariance-robustness-v2` | K(α,h,S) surface, portfolio firmness, diversification, eras, holdout |
| `flex_decomposition` | `flex_decomposition.py`, `flex_decomposition_deliverables.py` | `2026-09-15_flexibility-decomposition` | the scope → persistence → coincidence identity (E0 substrate, E1 chain, E2 Shapley, E3 bootstrap, E4 robustness) |
| `a7_m100_blind` | `m100_validation_v2.py` | `2026-09-15_m100-blind-crossmonth` | blind cross-month M100 validation: parameters locked on one month, applied unseen to others; per-month coverage |
| `a7_validation`, `a7b_reanchored_*` | `a7_validation_plot.py`; stages 1+5 with two re-anchored configs | v1.0 inputs (unchanged; measurement-based) | model anchors vs published measurements; sensitivity to measured-level re-anchoring |
| `a8_screening` | `a8_headroom.py` | `2026-09-18_a8-cap25` | illustrative MISO South peak-cap screen at a **25% cluster-power cap** (Nature Energy), three facility tiers |
| `coincidence_controls` | `coincidence_controls.py`, `reviewB_b2_pind_robustness.py` | `2026-09-18_chatgpt3-coincidence` + `2026-09-18_review-B-experiments` | negative controls for the coincidence penalty (E1 truncation-inactive, E2 single-cluster, E3 phase-randomized surrogates) and lower-tail robustness |
| `tail_events` | `tail_event_dedup.py` | `2026-09-18_chatgpt3-tail` | de-duplicated trough events (the 99%/24h count reduces to a few independent events → historical description), forward-split time audit, deliverable-derating trade-off |
| `fig7a_portfolio` | `fig7a_portfolio_scan.py` | `2026-09-18_fig7a-portfolio` | four-hour availability ratio vs clusters pooled (observed, independent benchmark, nested bootstrap ribbon) |
| `review_b` | `reviewB_b1_backtest_independent_calib.py`, `reviewB_b3_m100_bin_propagation.py` | `2026-09-18_review-B-experiments` | independent-calibration backtest; M100 utilisation-bin error propagation |
| `review_c` | `reviewC_c1_facility_scope_propagation.py`, `reviewC_c3_cluster_set_consistency.py` | `2026-09-18_review-C-experiments` | facility-scope propagation; cluster-set consistency |
| `r1_realizable` | `r1_realizable_sweep.py` | `2026-09-04_review-followup` | realizable fraction q, K∝q homogeneity, rebound/recovery vs q |
| `r2_rolling_origin` | `r2_rolling_origin.py` | `2026-09-04_review-followup` | rolling-origin validation of K, derating factors |
| `r4_covariance` | `r4_covariance_decomposition.py`, `covariance_robustness_v2.py` | `2026-09-09_covariance-robustness-v2` | exact covariance decomposition, matched four-cluster portfolio |
| `envelope_dualband` | `fig2_envelope_dualband.py` | `2026-09-18_chatgpt3-presentation` | order-statistic index behind the dual-band envelope figure (Fig. 2) |

`a6_tail_scalar/` and `a6_required_share/` are retained from v1.0 for traceability; their
tail-calibration and required-share content is superseded by the covariance surface
(`a6_availability`, `r4_covariance`) and the realizable-fraction sweep (`r1_realizable`).
