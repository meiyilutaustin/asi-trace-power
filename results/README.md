# results/

One directory per pipeline stage, each holding the `summary.json` written by the script,
any small tables, and the figures. `headline.json` gathers every number quoted in the
paper, keyed by the stage it comes from; `_provenance` records the model conventions
(idle-retained boundary, central eligibility, online-inference floor).

| directory | script | note |
|---|---|---|
| `a1_load` | `a1_load_reconstruction.py` | facility band, growth, daily profile, components |
| `a2_scaling` | `a2_fluctuation_scaling.py` | Taylor exponents, synchrony under three residual definitions |
| `a3_join_audit`, `a3_scheduler` | `a3_join_audit.py`, `a3_battery_inversion.py` | join coverage by day/label; difference process and backlog replay (post-coverage window) |
| `a5_envelope` | `a5_envelope.py` | envelope, boundaries, eligibility mappings, MC bands |
| `a6_availability`, `a6_tail_scalar`, `a6_required_share` | `a6_*.py` | K surface (13-cluster portfolio), tail-calibrated scalar, required-share grid (15-cluster fleet series; do not mix with the 13-cluster table) |
| `a7_validation`, `a7b_reanchored_*` | `a7_validation_plot.py`; stages 1+5 with the two re-anchored configs | anchors vs measurements; sensitivity of the headline to measured-level re-anchoring |
| `a8_screening` | `a8_headroom.py` | illustrative MISO South peak-cap screening, three facility tiers |
| `a9_formulations` | `a9_formulations.py` | published formulations on a common boundary, repeat-event budget, host add-on |
| `a10_jensen_online`, `a11_intrahour` | `a10_*.py`, `a11_*.py` | Jensen bound; within-hour minima |
| `r1_realizable`, `r2_rolling_origin`, `r4_covariance` | `r1_*.py`, `r2_*.py`, `r4_*.py` | realizable fraction q; rolling-origin validation; covariance decomposition |
