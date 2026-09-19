#!/usr/bin/env python
"""B1 (reviewer 2026-09-18 §2): rolling-origin backtest of the envelope K with an
INDEPENDENT calibration interval for the derating factor.

The prior backtest (src/r2_rolling_origin.py, runs/2026-09-04_review-followup/r2)
chose the derating factor d on the SAME pooled test windows it then scored, which
is retrospective calibration. The reviewer asks for a disjoint calibration and
test interval so the derating is a genuine forward rule, and for per-start
coverage (not just the cross-period mean).

This script:
  - reproduces the rolling-origin windows (8-week train, 4-week test, weekly step),
  - splits the origins CHRONOLOGICALLY: earlier origins = calibration, later
    origins = test (a real time-forward split); also runs the reverse split as a
    symmetry check,
  - on the calibration origins only, picks the smallest derating d so pooled
    calibration coverage >= target,
  - applies that fixed d to the held-out test origins and reports:
      * pooled test coverage with d=1 (no derating) and with the calibrated d,
      * per-start test coverage (min / worst start), fraction of starts meeting
        target, mean/worst shortfall depth.
Units: K_abs (MW fixed last quarter). alpha in {0.95, 0.90}, h in {1,4,24}.

Reuses the exact envelope series and kval definition of r2_rolling_origin.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

RERUN = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-09-04_rerun-p0"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/2026-09-18_review-B-experiments/b1_backtest"
os.makedirs(OUT, exist_ok=True)
HS = [1, 4, 24]
ALPHAS = [0.95, 0.90]
TRAIN_W = 8          # weeks (the headline setting in the prior backtest)
TEST_W = 4           # weeks
STEP_W = 1
W = 168


def rmin(s, h):
    m = sliding_window_view(s, h).min(axis=1) if h > 1 else s
    return m[~np.isnan(m)]


def kval(s, h, a):
    m = rmin(s, h)
    return float(np.quantile(m, 1 - a)) if len(m) else np.nan


def collect_windows(C, nt):
    """One record per (origin, h): train-K and the test running-min arrays."""
    recs = []
    o = TRAIN_W * W
    while o + TEST_W * W <= nt:
        tr = C[o - TRAIN_W * W:o]
        te = C[o:o + TEST_W * W]
        if np.isfinite(tr).sum() >= 0.5 * len(tr) and np.isfinite(te).sum() >= 0.5 * len(te):
            rec = {"origin_day": o // 24, "mean_train": float(np.nanmean(tr)),
                   "mean_test": float(np.nanmean(te)), "mt": {}}
            for h in HS:
                mt = rmin(te, h)
                rec["mt"][h] = mt if len(mt) >= 10 else None
                for a in ALPHAS:
                    rec[f"K_{h}_{a}"] = kval(tr, h, a)
            recs.append(rec)
        o += STEP_W * W
    return recs


def pooled_coverage(recs, h, a, d):
    """Coverage pooled across the given origins at derating d (per-start list too)."""
    per_start = []
    for r in recs:
        mt = r["mt"][h]
        if mt is None:
            continue
        k = d * r[f"K_{h}_{a}"]
        per_start.append(float(np.mean(mt >= k)))
    return per_start


def calibrate_d(recs, h, a):
    """Smallest d with pooled (concatenated) calibration coverage >= a."""
    # pool all test running-minima and their per-window K, weight by hours
    mts, ks = [], []
    for r in recs:
        mt = r["mt"][h]
        if mt is None:
            continue
        mts.append(mt)
        ks.append(np.full(len(mt), r[f"K_{h}_{a}"]))
    if not mts:
        return np.nan
    mt = np.concatenate(mts)
    k = np.concatenate(ks)
    lo, hi = 0.3, 1.5
    for _ in range(50):
        d = 0.5 * (lo + hi)
        if np.mean(mt >= d * k) >= a:
            lo = d
        else:
            hi = d
        if hi - lo < 1e-4:
            break
    return float(lo)


def evaluate(recs_cal, recs_te, h, a, tag):
    d = calibrate_d(recs_cal, h, a)
    cov_te_d1 = pooled_coverage(recs_te, h, a, 1.0)
    cov_te_d = pooled_coverage(recs_te, h, a, d)
    cal_cov_d1 = pooled_coverage(recs_cal, h, a, 1.0)

    def stats(per_start):
        v = np.array(per_start, float)
        return {"n_starts": int(len(v)),
                "coverage_mean": float(v.mean()) if len(v) else None,
                "coverage_worst_start": float(v.min()) if len(v) else None,
                "coverage_p10_start": float(np.quantile(v, 0.1)) if len(v) else None,
                "frac_starts_meeting_target": float((v >= a).mean()) if len(v) else None}
    return {"split": tag, "h": h, "alpha": a,
            "derating_calibrated": d,
            "calibration": stats(cal_cov_d1),
            "test_no_derating": stats(cov_te_d1),
            "test_with_calibrated_derating": stats(cov_te_d),
            "n_test_starts": len(cov_te_d)}


def main():
    env = pd.read_parquet(os.path.join(RERUN, "a5", "envelope_hourly.parquet"))
    good = env["good"].to_numpy().astype(bool)
    C = np.where(good, env["curtail_idle_retained"].to_numpy(), np.nan)
    nt = len(C)
    recs = collect_windows(C, nt)
    n = len(recs)
    mid = n // 2
    splits = {
        "forward_earlyCal_lateTest": (recs[:mid], recs[mid:]),
        "reverse_lateCal_earlyTest": (recs[mid:], recs[:mid]),
    }
    out = {"inputs": {"hours": int(good.sum()), "mean_mw": float(np.nanmean(C)),
                      "n_origins_total": n, "train_weeks": TRAIN_W, "test_weeks": TEST_W,
                      "origin_days": [r["origin_day"] for r in recs]},
           "note": ("Derating d is fitted ONLY on the calibration origins and then "
                    "applied unchanged to the disjoint test origins. Coverage is "
                    "reported per start (worst start, fraction meeting target), not "
                    "only as the cross-period mean.")}
    results = []
    for tag, (cal, te) in splits.items():
        for h in HS:
            for a in ALPHAS:
                results.append(evaluate(cal, te, h, a, tag))
    out["results"] = results
    # also: whole-record (in-sample) K and its retrospective derating for reference
    out["whole_record_K_mw"] = {f"h{h}_a{a}": kval(C, h, a) for h in HS for a in ALPHAS}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # readable console table
    print(f"origins={n}  calib={mid}  test={n-mid}")
    for r in results:
        if r["split"].startswith("forward") and r["alpha"] == 0.95:
            t1 = r["test_no_derating"]; t2 = r["test_with_calibrated_derating"]
            print(f"h={r['h']:2d} a=0.95  d={r['derating_calibrated']:.3f}  "
                  f"test cov (no derate) mean={t1['coverage_mean']:.3f} worst={t1['coverage_worst_start']:.3f} "
                  f"meet={t1['frac_starts_meeting_target']:.2f} | "
                  f"with derate mean={t2['coverage_mean']:.3f} worst={t2['coverage_worst_start']:.3f} "
                  f"meet={t2['frac_starts_meeting_target']:.2f}")
    print("wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
