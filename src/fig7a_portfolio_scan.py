#!/usr/bin/env python3
"""Fig 7(a) recompute: availability ratio vs portfolio size (方案 B).

Re-analysis on the EXISTING E0 substrate (13 Alibaba clusters, idle-retained
LP-active, central eligibility, GPU-side workload MW basis) — no new power-model
fit, no new simulation. Identical substrate/definitions as
src/a6_accreditation.py and src/flex_decomposition.py.

For each portfolio size n = 1..13:
  - subsets: enumerate all C(13,n) when <= N_CAP, else N_CAP random subsets;
  - per subset: P_obs = P(sum_i C_i, T, a);  P_indep = decorrelated-independent
    firm power (null_indep_v2 / kval_decorrelated with n_shift=N_SHIFT);
  - firm_obs[n]   = mean_subsets(P_obs)   / mean_subsets(mean_sub)   (ratio of means, == a6)
    firm_indep[n] = mean_subsets(P_indep) / mean_subsets(mean_sub)
Horizons T in {1,4,24}. Uncertainty ribbon for the observed curve = nested
top-n 168 h block bootstrap P5/P95, reused from a6 (boot_firmness_nested_h4_a95).

P(x,T,a) = (1-alpha) quantile of the T-hour running minimum of x.
"""
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import flex_decomposition as F  # noqa: E402  (build_substrate, kval, trend_of)

AGG = REPO / "runs/2026-09-02_a1-load-reconstruction/pod_hourly_agg.parquet"
CFG = REPO / "configs/power_curves.yaml"
A6 = REPO / "runs/2026-09-04_rerun-p0/a6/summary.json"
OUT = Path(__file__).resolve().parent

HORIZONS = (1, 4, 24)
ALPHA = 0.95
N_CAP = 200          # enumerate all subsets when C(13,n) <= N_CAP, else sample N_CAP
N_SHIFT = 200        # circular-shift draws per subset for the independent benchmark
SEED = 20260918


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    C, A, Wf, good, meta = F.build_substrate(str(AGG), str(CFG))
    N = C.shape[0]
    assert N == 13, f"expected 13 clusters, got {N}"

    # precompute per-cluster trend + residual ONCE (trend_of is per-row, so a
    # subset's trend is exactly the subset of the full trend matrix).
    tr = F.trend_of(C)                 # (13, nt)
    res = C - tr                       # (13, nt); NaN at ~good columns
    nt = C.shape[1]
    rng = np.random.default_rng(SEED)

    def indep_firm(idx, shifts):
        """decorrelated independent-summed firm power, averaged over N_SHIFT draws
        (== null_indep_v2 / kval_decorrelated for this subset)."""
        vals = []
        for row in shifts:               # row: one shift per cluster in idx
            s = np.zeros(nt)
            for j, i in enumerate(idx):
                s += np.roll(res[i], row[j]) + tr[i]
            for_h = {}
            m = np.maximum(s, 0.0)
            vals.append(m)
        # evaluate all horizons on the shared decorrelated series set
        out = {}
        for T in HORIZONS:
            out[T] = float(np.nanmean([F.kval(m, T, ALPHA) for m in vals]))
        return out

    results = {T: {} for T in HORIZONS}
    for n in range(1, N + 1):
        total = comb(N, n)
        if total <= N_CAP:
            combos = list(itertools.combinations(range(N), n))
        else:
            seen, combos = set(), []
            while len(combos) < N_CAP:
                c = tuple(sorted(rng.choice(N, n, replace=False)))
                if c not in seen:
                    seen.add(c)
                    combos.append(c)
        # accumulators per horizon
        acc = {T: {"P_obs": [], "P_indep": []} for T in HORIZONS}
        means = []
        for cmb in combos:
            idx = list(cmb)
            agg = np.where(good, np.nansum(C[idx], axis=0), np.nan)
            means.append(float(np.nanmean(agg)))
            shifts = rng.integers(0, nt, size=(N_SHIFT, n))
            ind = indep_firm(idx, shifts)
            for T in HORIZONS:
                acc[T]["P_obs"].append(F.kval(agg, T, ALPHA))
                acc[T]["P_indep"].append(ind[T])
        mean_of_means = float(np.mean(means))
        for T in HORIZONS:
            po = float(np.mean(acc[T]["P_obs"]))
            pi = float(np.mean(acc[T]["P_indep"]))
            results[T][n] = {
                "firm_obs": po / mean_of_means,
                "firm_indep": pi / mean_of_means,
                "mean_P_obs_mw": po,
                "mean_P_indep_mw": pi,
                "mean_mw": mean_of_means,
                "n_subsets": len(combos),
                "enumerated": total <= N_CAP,
            }
        print(f"n={n:2d} subsets={len(combos):3d}  "
              f"obs(4h)={results[4][n]['firm_obs']:.4f}  "
              f"indep(4h)={results[4][n]['firm_indep']:.4f}", flush=True)

    # reuse a6 nested-top-n 168h block bootstrap band for the observed ribbon
    a6 = json.loads(A6.read_text())
    boot = a6["boot_firmness_nested_h4_a95"]
    ribbon = {int(k): {"p5": v["p5"], "p50": v["p50"], "p95": v["p95"], "base": v["base"]}
              for k, v in boot.items()}

    prov = {
        "script_sha256": sha256(__file__),
        "flex_decomposition_sha256": sha256(REPO / "src/flex_decomposition.py"),
        "powermodel_sha256": sha256(REPO / "src/powermodel.py"),
        "input_agg": str(AGG),
        "input_agg_sha256": sha256(AGG),
        "config_sha256": sha256(CFG),
        "a6_summary_sha256": sha256(A6),
        "n_shift": N_SHIFT, "n_cap": N_CAP, "alpha": ALPHA, "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {
        "meta": meta,
        "definition": "firm = mean_subsets(P(sum C_i,T,a)) / mean_subsets(mean eligible); "
                      "P = (1-a) quantile of T-hour running min. indep = decorrelated "
                      "independent-summed (n_shift=200). ratio-of-means == a6.",
        "horizons": list(HORIZONS), "alpha": ALPHA,
        "by_horizon": {f"h{T}": {str(n): results[T][n] for n in range(1, N + 1)}
                       for T in HORIZONS},
        "observed_ribbon_h4_nested_top_n_168h_block_bootstrap": {
            str(k): v for k, v in ribbon.items()},
        "provenance": prov,
    }
    (OUT / "portfolio_scan.json").write_text(json.dumps(payload, indent=2))
    print("wrote", OUT / "portfolio_scan.json")


if __name__ == "__main__":
    main()
