#!/usr/bin/env python
"""B2 (reviewer 2026-09-18 §4): lower-tail robustness of the envelope P and the
independent benchmark P^ind (not just the synchrony ratio R).

The prior covariance work (runs/2026-09-04_review-followup/r4) established that the
SYNCHRONY RATIO R (sync_ratio / excess covariance X) is robust to de-seasoning,
capacity-normalisation and stable-platform restriction. The reviewer notes that
the paper's headline lower-tail quantities are the envelope P(tau,alpha) and the
independent benchmark P^ind(tau,alpha) themselves, and asks whether those are
equally robust to:

  (1) a stable-capacity window (the raw trajectory's fleet demand grows ~39% end
      to end, so a free circular shift also re-aligns different growth phases),
  (2) a calendar-preserving randomization (shift residuals by whole-day / whole-
      week multiples, so hour-of-day and day-of-week phase are preserved),
  (3) detrend-then-restore vs shifting the whole (trended) series,
  (4) the choice of "average of the per-shift quantiles" vs a single quantile of
      the pooled decorrelated sample.

This script recomputes, on the exact E0 substrate (13 clusters, idle-retained
LP-active), the absolute MW values of:
  P^obs(tau,alpha) = kval(A)            (honest coincident aggregate; the envelope)
  P^ind(tau,alpha) = independent aggregate under each randomization scheme
  R = P^obs / P^ind                     (the coincidence factor)
for tau in {1,4,24} h at alpha=0.95, on the full record and on a stable-capacity
window, for every (randomization, aggregation) combination.

Reuses flex_decomposition.build_substrate / kval / trend_of so the numbers are
directly comparable to the manuscript's decomposition.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import flex_decomposition as fd  # noqa: E402

AGG = os.path.join(HERE, "..", "2026-09-02_a1-load-reconstruction", "pod_hourly_agg.parquet")
CFG = os.path.join(HERE, "..", "..", "configs", "power_curves.yaml")
OUT = os.path.join(HERE, "b2_pind_robustness")
os.makedirs(OUT, exist_ok=True)

HORIZONS = (1, 4, 24)
ALPHA = 0.95
NSHIFT = 200
SEED = 20260918


def independent_samples(C, T, alpha, rng, n_shift, mode):
    """Return the list of decorrelated-aggregate T-hour running-minima arrays
    (one per shift draw), under randomization `mode`:
      free_detrend_restore : detrend, roll residual by any offset, re-add trend
      calendar             : detrend, roll residual by a multiple of 24 h, re-add
      week                 : detrend, roll residual by a multiple of 168 h, re-add
      free_whole_series    : roll the whole (trended) series by any offset
    """
    nt = C.shape[1]
    if mode == "free_whole_series":
        tr = np.zeros_like(C)
        res = C
    else:
        tr = fd.trend_of(C)
        res = C - tr
    mins = []
    for _ in range(n_shift):
        s = np.zeros(nt)
        for i in range(C.shape[0]):
            if mode == "calendar":
                off = int(rng.integers(0, nt // 24)) * 24
            elif mode == "week":
                off = int(rng.integers(0, max(nt // 168, 1))) * 168
            else:
                off = int(rng.integers(0, nt))
            s += np.roll(res[i], off) + (tr[i] if mode != "free_whole_series" else 0.0)
        s = np.maximum(s, 0.0)
        mins.append(fd.rolling_min(s, T))
    return mins


def pind(C, T, alpha, rng, n_shift, mode, agg):
    mins = independent_samples(C, T, alpha, rng, n_shift, mode)
    if agg == "avg_of_quantiles":
        return float(np.mean([np.quantile(m, 1 - alpha) for m in mins if len(m)]))
    elif agg == "pooled_joint_quantile":
        pooled = np.concatenate([m for m in mins if len(m)])
        return float(np.quantile(pooled, 1 - alpha))
    raise ValueError(agg)


def stable_capacity_mask(A, band=0.10):
    """Hours whose 168 h fleet-eligible trend lies within +/- band of the median
    trend: a flat-capacity window that excludes the growth ramp."""
    tr = fd.trend_of(A[None, :])[0]
    med = np.nanmedian(tr)
    return np.abs(tr - med) <= band * med


def run_window(C, A, tag):
    rng = np.random.default_rng(SEED)
    res = {"window": tag, "n_hours": int(np.isfinite(A).sum()),
           "mean_eligible_mw": float(np.nanmean(A))}
    # honest envelope (does not depend on randomization)
    res["P_obs_mw"] = {f"h{T}": float(fd.kval(A, T, ALPHA)) for T in HORIZONS}
    modes = ["free_detrend_restore", "calendar", "week", "free_whole_series"]
    aggs = ["avg_of_quantiles", "pooled_joint_quantile"]
    res["P_ind_mw"] = {}
    res["coincidence_factor_R"] = {}
    for mode in modes:
        for agg in aggs:
            key = f"{mode}__{agg}"
            pv, rv = {}, {}
            for T in HORIZONS:
                p = pind(C, T, ALPHA, np.random.default_rng(SEED + T), NSHIFT, mode, agg)
                pv[f"h{T}"] = p
                rv[f"h{T}"] = float(fd.kval(A, T, ALPHA) / p) if p else float("nan")
            res["P_ind_mw"][key] = pv
            res["coincidence_factor_R"][key] = rv
    return res


def main():
    C, A, Wf, good, meta = fd.build_substrate(AGG, CFG)
    out = {"substrate": {k: meta[k] for k in ("n_clusters", "valid_hours",
            "mean_eligible_mw", "measured_scope_share")},
           "alpha": ALPHA, "horizons": list(HORIZONS), "n_shift": NSHIFT,
           "note": ("P_obs is the honest coincident envelope; P_ind is the "
                    "independent (decorrelated) benchmark. R = P_obs/P_ind is the "
                    "coincidence factor (<1 = coincident low periods reduce firm "
                    "power). 'calendar'/'week' preserve diurnal/weekly phase; "
                    "'pooled_joint_quantile' takes one quantile of the pooled "
                    "decorrelated sample instead of averaging per-draw quantiles.")}
    # full record
    full = run_window(C, A, "full_record")
    # stable-capacity window
    mask = stable_capacity_mask(A, band=0.10)
    Cs = C.copy(); Cs[:, ~mask] = np.nan
    As = A.copy(); As[~mask] = np.nan
    stable = run_window(Cs, As, "stable_capacity_trendband_0.10")
    out["results"] = {"full_record": full, "stable_capacity": stable}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # console summary
    def show(r):
        print(f"\n[{r['window']}]  n={r['n_hours']} h  mean eligible {r['mean_eligible_mw']:.3f} MW")
        print("  P_obs (MW):", {k: round(v, 3) for k, v in r["P_obs_mw"].items()})
        for key in r["P_ind_mw"]:
            pv = r["P_ind_mw"][key]; rv = r["coincidence_factor_R"][key]
            print(f"  {key:40s} P_ind={ {k:round(v,3) for k,v in pv.items()} }  "
                  f"R={ {k:round(v,3) for k,v in rv.items()} }")
    show(full)
    show(stable)
    print("\nwrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
