#!/usr/bin/env python3
"""Post-process the canonical A6 summary into a 4 h/95% tail scalar.

This is an algebraic correction, not a new experiment.  A6's constant-share
baseline is

    K_const(h) = p_mean K_L(h),

where L is the workload-power series for the same retained 13-cluster
portfolio.  Calibrating p_tail so that p_tail K_L(4) = K_C(4) therefore gives

    p_tail / p_mean = 1 / [K_const(4) / K_C(4)].

Positive homogeneity then gives the corrected scalar/observed ratio at every
h directly from the saved A6 point estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from provenance import provenance


HORIZONS = (1, 4, 24)
ALPHA = 0.95
CALIBRATION_HOURS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def headline_key(hours: int) -> str:
    return f"h{hours}_a{ALPHA}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="canonical A6 summary.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plot-source", type=Path, required=True)
    parser.add_argument("--res-figure", type=Path, required=True)
    parser.add_argument("--paper-figure", type=Path, required=True)
    args = parser.parse_args()

    a6 = json.loads(args.input.read_text())
    if a6["n_clusters"] != 13:
        raise ValueError(f"expected canonical 13-cluster A6 portfolio, got {a6['n_clusters']}")

    mean_eligible = float(a6["fleet_mean_mw"]["curtail"])
    mean_workload = float(a6["fleet_mean_mw"]["workload"])
    mean_share = mean_eligible / mean_workload

    by_horizon = {}
    for hours in HORIZONS:
        row = a6["headline_fleet_curtail"][headline_key(hours)]
        observed = float(row["k_obs_mw"]["workload"])
        mean_scalar = float(row["k_const_mw"])
        mean_ratio = float(row["const_over_obs"])
        if abs(mean_scalar / observed - mean_ratio) > 1e-12:
            raise ValueError(f"A6 constant-share ratio does not close at h={hours}")
        by_horizon[hours] = {
            "observed_k_mw": observed,
            "mean_calibrated_scalar_k_mw": mean_scalar,
            "mean_calibrated_scalar_over_observed": mean_ratio,
            "implied_workload_k_mw": mean_scalar / mean_share,
        }

    calibration = by_horizon[CALIBRATION_HOURS]
    tail_share = calibration["observed_k_mw"] / calibration["implied_workload_k_mw"]
    scale_from_mean_scalar = tail_share / mean_share

    for hours in HORIZONS:
        row = by_horizon[hours]
        tail_k = tail_share * row["implied_workload_k_mw"]
        tail_ratio = tail_k / row["observed_k_mw"]
        row["tail_calibrated_scalar_k_mw"] = tail_k
        row["tail_calibrated_scalar_over_observed"] = tail_ratio

    if abs(by_horizon[CALIBRATION_HOURS]["tail_calibrated_scalar_over_observed"] - 1.0) > 1e-12:
        raise ValueError("4 h ratio is not one after by-construction calibration")

    plot_text = args.plot_source.read_text()
    plot_block = plot_text.split("# ---- A6: K vs duration", 1)[1].split(
        "# ---- A6: portfolio", 1)[0]
    tail_tokens = ("tail-calibrated", "tail calibrated", "tail_calibrated", "p_tail")
    tail_scalar_in_plot = any(token in plot_block for token in tail_tokens)
    mean_scalar_in_plot = "constant share calibrated to the mean" in plot_block
    res_figure_hash = sha256(args.res_figure)
    paper_figure_hash = sha256(args.paper_figure)

    result = {
        "calculation": "A6 13-cluster workload scalar calibrated by construction to K(0.95, 4 h)",
        "source_a6_summary": str(args.input),
        "scope": {
            "n_clusters": int(a6["n_clusters"]),
            "included_cluster_ids": list(a6["cluster_mean_curtail_mw"].keys()),
            "excluded_clusters": a6["excluded_clusters"],
            "minimum_cluster_mean_eligible_mw": float(a6["settings"]["min_cluster_mw"]),
            "eligibility": a6["settings"]["eligibility"],
            "preemption_boundary": a6["settings"]["boundary"],
            "power_boundary": "GPU-side workload power; facility-marginal ratio is numerically identical at response multiplier 1.0",
            "scalar_denominator": "the same 13-cluster A6 GPU-side workload-power series L(t)",
            "mean_eligible_mw": mean_eligible,
            "mean_workload_mw": mean_workload,
            "mean_eligible_share_of_workload": mean_share,
        },
        "product": {
            "alpha": ALPHA,
            "calibration_duration_hours": CALIBRATION_HOURS,
            "definition": "K(alpha,h) is the (1-alpha) quantile of h-hour running minima",
        },
        "formula": {
            "mean_scalar": "K_mean(h) = p_mean K_L(h)",
            "tail_share": "p_tail = K_C(4,0.95) / K_L(4,0.95)",
            "tail_ratio": "K_tail(h)/K_C(h) = [K_mean(h)/K_C(h)] / [K_mean(4)/K_C(4)]",
        },
        "calibration": {
            "tail_share_of_workload": tail_share,
            "tail_share_of_workload_pct": 100.0 * tail_share,
            "tail_share_over_mean_share": scale_from_mean_scalar,
            "four_hour_closure_error_mw": (
                by_horizon[CALIBRATION_HOURS]["tail_calibrated_scalar_k_mw"]
                - by_horizon[CALIBRATION_HOURS]["observed_k_mw"]
            ),
        },
        "by_horizon": {str(hours): by_horizon[hours] for hours in HORIZONS},
        "stale_table_values": {
            "reported_ratios_h1_h4_h24": [0.790, 0.857, 1.019],
            "status": "invalid as a by-construction 4 h calibration because the 4 h ratio is not 1",
            "replacement_ratios_h1_h4_h24": [
                by_horizon[hours]["tail_calibrated_scalar_over_observed"]
                for hours in HORIZONS
            ],
        },
        "k_surface_figure_audit": {
            "plot_source": str(args.plot_source),
            "res_figure": str(args.res_figure),
            "paper_figure": str(args.paper_figure),
            "res_figure_sha256": res_figure_hash,
            "paper_figure_sha256": paper_figure_hash,
            "figures_identical": res_figure_hash == paper_figure_hash,
            "mean_calibrated_scalar_present": mean_scalar_in_plot,
            "tail_calibrated_scalar_present": tail_scalar_in_plot,
            "regeneration_required_for_this_correction": tail_scalar_in_plot,
            "reason": (
                "The plot reads A6 fleet.k_const and labels it as calibrated to the mean; "
                "it does not display the stale tail-calibrated table series."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "summary.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    prov = provenance()
    prov.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_type": "deterministic algebraic post-processing of saved A6 point estimates",
        "executed_script": str(Path(__file__).resolve()),
        "executed_script_sha256": sha256(Path(__file__).resolve()),
        "command_cwd": os.getcwd(),
        "inputs": {
            "a6_summary": {"path": str(args.input), "sha256": sha256(args.input)},
            "plot_source": {"path": str(args.plot_source), "sha256": sha256(args.plot_source)},
            "a6_res_figure": {"path": str(args.res_figure), "sha256": res_figure_hash},
            "paper_figure": {"path": str(args.paper_figure), "sha256": paper_figure_hash},
        },
        "outputs": ["README.md", "summary.json", "provenance.json"],
        "compute_note": "No raw-data scan, Monte Carlo, bootstrap, or figure regeneration was run.",
    })
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    main()
