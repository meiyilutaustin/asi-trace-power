#!/usr/bin/env python
"""Assemble results/headline.json from the per-stage result files.

Every value in the headline is copied verbatim from a stage's ``summary.json``
(or equivalent), so the headline never introduces a number that is not already
in a shipped result file. ``_provenance.sources`` records, per block, the file
each subtree was taken from and the canonical run it came from (see
results/README.md). Run from the repository root:

    python src/build_headline.py
"""
import json
import os

R = "results"


def J(*p):
    with open(os.path.join(R, *p)) as f:
        return json.load(f)


def sub(d, *path):
    for k in path:
        d = d[k]
    return d


# --- load stage outputs ---
a1 = J("a1_load", "summary.json")
a2 = J("a2_scaling", "descriptive_beta.json")
a6 = J("a6_availability", "summary.json")
e0 = J("flex_decomposition", "E0_substrate.json")
e1 = J("flex_decomposition", "E1_chain.json")
a8 = J("a8_screening", "summary.json")
f7 = J("fig7a_portfolio", "portfolio_scan.json")
coinc = J("coincidence_controls", "e1e2e3.json")
tail = J("tail_events", "e4e5e6.json")
r1 = J("r1_realizable", "summary.json")
r2 = J("r2_rolling_origin", "summary.json")
m100 = J("a7_m100_blind", "coverage.json")
e8 = J("envelope_dualband", "e8_index.json")

sources = {
    "fleet_mw": "results/a1_load/summary.json  (run 2026-09-04_rerun-p0)",
    "synchrony": "results/a2_scaling/descriptive_beta.json  (run 2026-09-09_covariance-robustness-v2)",
    "scope_decomposition": "results/flex_decomposition/E0_substrate.json, E1_chain.json  (run 2026-09-15_flexibility-decomposition)",
    "curtail_K_surface": "results/a6_availability/summary.json  (run 2026-09-04_rerun-p0 + 2026-09-09_covariance-robustness-v2)",
    "portfolio_availability": "results/fig7a_portfolio/portfolio_scan.json  (run 2026-09-18_fig7a-portfolio)",
    "peak_cap_screen": "results/a8_screening/summary.json  (run 2026-09-18_a8-cap25, 25% cluster-power cap)",
    "coincidence_controls": "results/coincidence_controls/e1e2e3.json  (run 2026-09-18_chatgpt3-coincidence)",
    "tail_events": "results/tail_events/e4e5e6.json  (run 2026-09-18_chatgpt3-tail)",
    "realizable": "results/r1_realizable/summary.json  (run 2026-09-04_review-followup)",
    "rolling_origin": "results/r2_rolling_origin/summary.json  (run 2026-09-04_review-followup)",
    "m100_blind_crossmonth": "results/a7_m100_blind/coverage.json  (run 2026-09-15_m100-blind-crossmonth)",
    "envelope_dualband": "results/envelope_dualband/e8_index.json  (run 2026-09-18_chatgpt3-presentation)",
}

headline = {
    "_provenance": {
        "note": (
            "All MW are GPU-side workload power unless a facility/IT boundary is named. "
            "Curtailment uses the idle-retained boundary and central eligibility. "
            "Every value here is copied verbatim from the stage file named in 'sources'; "
            "this index restates nothing that is not in a shipped result file."
        ),
        "boundary": e0.get("boundary"),
        "eligibility": e0.get("eligibility"),
        "sources": sources,
    },
    "fleet_mw": {
        "mean_mw": a1["mean_mw"],
        "mean_it_mw_p50": a1["mean_it_mw_p50"],
        "peak_mw_p50": a1["peak_mw_p50"],
        "load_factor_p50": a1["load_factor_p50"],
        "growth": a1["growth"],
        "avg_daily_amplitude_pct_of_mean": a1["avg_daily_amplitude_pct_of_mean"],
        "daily_profile_peak_hour": a1["daily_profile_peak_hour"],
        "daily_profile_trough_hour": a1["daily_profile_trough_hour"],
        "autocorr_24h": a1["autocorr_24h"],
        "autocorr_7d": a1["autocorr_7d"],
        "component_share_pct_base": a1["component_share_pct_base"],
        "online_floor_share_of_it_pct_base": a1["online_floor_share_of_it_pct_base"],
    },
    "synchrony": a2,
    "scope_decomposition": {
        "measured_scope_share": e0.get("measured_scope_share"),
        "mean_eligible_mw": e0.get("mean_eligible_mw"),
        "mean_workload_mw": e0.get("mean_workload_mw"),
        "n_clusters": e0.get("n_clusters"),
        "chain_by_s0": e1["by_s0"],
    },
    "curtail_K_surface": {
        "headline_fleet_curtail": a6["headline_fleet_curtail"],
        "curtail_share_of_workload_pct": a6["curtail_share_of_workload_pct"],
        "portfolio_firmness_h4_a95": a6["portfolio_firmness_h4_a95"],
        "diversification_gain_h4_a95": a6.get("diversification_gain_h4_a95"),
        "n_clusters": a6["n_clusters"],
    },
    "portfolio_availability": {
        "definition": f7.get("definition"),
        "by_horizon": f7["by_horizon"],
        "observed_ribbon_h4": f7.get("observed_ribbon_h4_nested_top_n_168h_block_bootstrap"),
    },
    "peak_cap_screen": {
        "rule": a8.get("rule"),
        "tiers": a8["tiers"],
    },
    "coincidence_controls": {
        "E1_fidelity": coinc.get("E1_fidelity"),
        "E2_single_cluster": coinc.get("E2_single_cluster"),
        "E3_synthetic_independent": coinc.get("E3_synthetic_independent"),
    },
    "tail_events": {
        "E4_trough_events": tail.get("E4_trough_events"),
        "E5_time_ranges": tail.get("E5_time_ranges"),
        "E6_derating_le1": tail.get("E6_derating_le1"),
    },
    "realizable": r1["headline"],
    "rolling_origin": {
        "by_setting": r2["by_setting"],
        "derating_trainweeks8": r2.get("derating_trainweeks8"),
        "interpretation": r2.get("interpretation"),
    },
    "m100_blind_crossmonth": m100,
    "envelope_dualband": e8,
}

out = os.path.join(R, "headline.json")
with open(out, "w") as f:
    json.dump(headline, f, indent=2)
print("wrote", out)
