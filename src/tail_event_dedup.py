#!/usr/bin/env python3
"""ChatGPT round-3 review, Q3 (tail rescue): E4 de-duplicated trough-event count,
E5 forward-split origin time ranges + overlap audit, E6 derating constrained to <=1
trade-off curve.

Re-analysis only. E4 uses the honest aggregate A on the canonical 13-cluster /
4,439 h substrate; E5/E6 reproduce the b1 rolling-origin backtest envelope series
(runs/2026-09-18_review-B-experiments/b1_backtest_independent_calib.py:
8-week train, 4-week test, weekly step, K = (1-alpha) quantile of the T-hour
running min of the training window).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import flex_decomposition as fx  # noqa: E402

AGG = os.path.join(REPO, "runs/2026-09-02_a1-load-reconstruction/pod_hourly_agg.parquet")
CFG = os.path.join(REPO, "configs/power_curves.yaml")
RERUN = os.path.join(REPO, "runs/2026-09-04_rerun-p0")
HORIZONS = (1, 4, 24)
W = 168
TRAIN_W, TEST_W, STEP_W = 8, 4, 1


# --------------------------------------------------------------------------- E4
def e4_trough_events(A):
    """Merge contiguous/overlapping sub-threshold T-hour running-min windows into
    distinct trough EVENTS. Report event count, durations, and week-spread at
    alpha in {0.95, 0.99} for T in {1,4,24}."""
    out = {}
    n = len(A)
    for T in HORIZONS:
        # running-min windows; index t covers calendar hours [t, t+T-1]
        if T == 1:
            m = np.asarray(A, float)
            starts_all = np.arange(n)
        else:
            m = sliding_window_view(np.asarray(A, float), T).min(axis=1)
            starts_all = np.arange(len(m))
        finite = ~np.isnan(m)
        m_f = m[finite]
        starts_f = starts_all[finite]
        Nwin = len(m_f)
        for alpha in (0.95, 0.99):
            thr = float(np.quantile(m_f, 1 - alpha))
            sub = m_f <= thr
            sub_starts = starts_f[sub]                 # window start hours <= threshold
            # each window covers [s, s+T-1]; merge overlapping/adjacent intervals
            events = []
            for s in np.sort(sub_starts):
                lo, hi = int(s), int(s + T - 1)
                if events and lo <= events[-1][1] + 1:
                    events[-1][1] = max(events[-1][1], hi)
                else:
                    events.append([lo, hi])
            durations = [hi - lo + 1 for lo, hi in events]
            weeks = sorted({lo // W for lo, hi in events})
            out[f"T{T}_a{alpha}"] = dict(
                threshold_mw=thr,
                n_subthreshold_windows=int(sub.sum()),
                n_windows_total=int(Nwin),
                order_stat_k=float((1 - alpha) * Nwin),
                n_events=len(events),
                event_durations_h=durations,
                total_event_hours=int(sum(durations)),
                n_distinct_weeks=len(weeks),
                weeks=weeks,
                events_hour_ranges=events,
                exploratory=bool(len(events) < 5),
            )
    return out


# ------------------------------------------------------------------- E5 / E6 shared
def envelope_series():
    env = pd.read_parquet(os.path.join(RERUN, "a5", "envelope_hourly.parquet"))
    good = env["good"].to_numpy().astype(bool)
    C = np.where(good, env["curtail_idle_retained"].to_numpy(), np.nan)
    return C, good


def rmin(s, h):
    m = sliding_window_view(s, h).min(axis=1) if h > 1 else np.asarray(s, float)
    return m


def kval(s, h, a):
    m = rmin(s, h)
    m = m[~np.isnan(m)]
    return float(np.quantile(m, 1 - a)) if len(m) else np.nan


def collect_windows(C, nt):
    recs = []
    o = TRAIN_W * W
    while o + TEST_W * W <= nt:
        tr = C[o - TRAIN_W * W:o]
        te = C[o:o + TEST_W * W]
        if np.isfinite(tr).sum() >= 0.5 * len(tr) and np.isfinite(te).sum() >= 0.5 * len(te):
            rec = {"origin_hour": o, "origin_day": o // 24,
                   "train_span_h": (o - TRAIN_W * W, o),
                   "test_span_h": (o, o + TEST_W * W), "mt": {}}
            for h in HORIZONS:
                mt = rmin(te, h); mt = mt[~np.isnan(mt)]
                rec["mt"][h] = mt if len(mt) >= 10 else None
                rec[f"K_{h}_0.95"] = kval(tr, h, 0.95)
            recs.append(rec)
        o += STEP_W * W
    return recs


# --------------------------------------------------------------------------- E5
def e5_time_ranges(recs):
    n = len(recs)
    mid = n // 2
    cal, te = recs[:mid], recs[mid:]

    def span_union(records, key):
        lo = min(r[key][0] for r in records)
        hi = max(r[key][1] for r in records)
        return lo, hi

    def fmt(h):
        return dict(hour=int(h), day=round(h / 24, 2), week=round(h / W, 2))

    # the *scored* hours are each origin's TEST window; calibration also fits on
    # the calibration origins' TEST windows -> compare those two unions.
    cal_test_lo, cal_test_hi = span_union(cal, "test_span_h")
    te_test_lo, te_test_hi = span_union(te, "test_span_h")
    overlap_lo, overlap_hi = max(cal_test_lo, te_test_lo), min(cal_test_hi, te_test_hi)
    overlap_h = max(0, overlap_hi - overlap_lo)

    # also: the training windows of the test origins vs calibration test windows
    te_train_lo, te_train_hi = span_union(te, "train_span_h")
    tr_ov_lo, tr_ov_hi = max(cal_test_lo, te_train_lo), min(cal_test_hi, te_train_hi)
    tr_overlap_h = max(0, tr_ov_hi - tr_ov_lo)

    return dict(
        n_origins=n, n_calibration=mid, n_test=n - mid,
        origins=[dict(origin_day=r["origin_day"],
                      train_days=[round(r["train_span_h"][0] / 24, 1), round(r["train_span_h"][1] / 24, 1)],
                      test_days=[round(r["test_span_h"][0] / 24, 1), round(r["test_span_h"][1] / 24, 1)])
                 for r in recs],
        calibration_test_window_union_days=[round(cal_test_lo / 24, 1), round(cal_test_hi / 24, 1)],
        test_test_window_union_days=[round(te_test_lo / 24, 1), round(te_test_hi / 24, 1)],
        scored_hours_overlap_h=int(overlap_h),
        scored_hours_overlap_days=round(overlap_h / 24, 1),
        scored_hours_overlap_weeks=round(overlap_h / W, 2),
        test_train_vs_calib_test_overlap_days=round(tr_overlap_h / 24, 1),
        calendar_convention="trace-relative days from trace start (day 0); absolute anchor not "
                            "required to prove disjointness — hour indices are exact.",
        note=("Forward split calibrates the derating on the earlier half of origins "
              "and scores the later half. The scored test windows of the two halves "
              "OVERLAP by the reported amount because 8+4-week origins stepped weekly "
              "share calendar hours. Reported transparently per reviewer line 88."),
    )


# --------------------------------------------------------------------------- E6
def e6_derating_le1(recs):
    """Trade-off curve: for d in [0.5, 1.0], pooled test coverage vs deliverable
    MW (= d * mean train-K), at alpha=0.95. Current forward rule needs d>1 to try
    to restore coverage; a deliverable rule must have d<=1 and accept lower coverage
    at each horizon. Uses the FORWARD test origins (held-out later half)."""
    n = len(recs); mid = n // 2
    te = recs[mid:]
    grid = np.round(np.arange(0.50, 1.001, 0.05), 3)
    out = {}
    for h in HORIZONS:
        mean_K = float(np.nanmean([r[f"K_{h}_0.95"] for r in te]))
        curve = []
        for d in grid:
            covs, mts, ks = [], [], []
            for r in te:
                mt = r["mt"][h]
                if mt is None:
                    continue
                k = d * r[f"K_{h}_0.95"]
                covs.append(float(np.mean(mt >= k)))
                mts.append(mt); ks.append(np.full(len(mt), k))
            mt_all = np.concatenate(mts); k_all = np.concatenate(ks)
            curve.append(dict(
                d=float(d),
                deliverable_mw=float(d * mean_K),
                pooled_coverage=float(np.mean(mt_all >= k_all)),
                per_start_coverage_mean=float(np.mean(covs)),
                per_start_coverage_worst=float(np.min(covs)),
                frac_starts_meeting_target=float(np.mean(np.array(covs) >= 0.95)),
            ))
        out[f"T{h}h"] = dict(mean_train_K_mw=mean_K, curve=curve)
    return out


def main():
    t0 = time.time()
    C, A, Wf, good, meta = fx.build_substrate(AGG, CFG)
    env_C, env_good = envelope_series()
    recs = collect_windows(env_C, len(env_C))

    out = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        substrate=dict(n_clusters=meta["n_clusters"], valid_hours=meta["valid_hours"]),
        E4_trough_events=e4_trough_events(A),
        E5_time_ranges=e5_time_ranges(recs),
        E6_derating_le1=e6_derating_le1(recs),
        runtime_sec=None,
    )
    out["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "e4e5e6.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("== E4 trough events (merged) ==")
    for T in HORIZONS:
        for a in (0.95, 0.99):
            r = out["E4_trough_events"][f"T{T}_a{a}"]
            print("  T=%2dh a=%.2f: %3d sub-windows -> %2d events, durations %s h, %d weeks%s"
                  % (T, a, r["n_subthreshold_windows"], r["n_events"],
                     r["event_durations_h"][:8], r["n_distinct_weeks"],
                     "  [EXPLORATORY <5 events]" if r["exploratory"] else ""))
    e5 = out["E5_time_ranges"]
    print("\n== E5 forward split ==")
    print("  %d origins: %d calib + %d test" % (e5["n_origins"], e5["n_calibration"], e5["n_test"]))
    print("  calib test-window union (days):", e5["calibration_test_window_union_days"])
    print("  test  test-window union (days):", e5["test_test_window_union_days"])
    print("  scored-hours overlap: %.1f days (%.2f weeks)"
          % (e5["scored_hours_overlap_days"], e5["scored_hours_overlap_weeks"]))
    print("\n== E6 derating<=1 trade-off (forward test, a=0.95) ==")
    for h in HORIZONS:
        print("  T=%2dh (mean K=%.3f MW):" % (h, out["E6_derating_le1"][f"T{h}h"]["mean_train_K_mw"]))
        for pt in out["E6_derating_le1"][f"T{h}h"]["curve"]:
            if pt["d"] in (0.6, 0.8, 0.9, 1.0):
                print("     d=%.2f  deliver=%.3f MW  pooled cov=%.3f  worst start=%.3f  meet=%.2f"
                      % (pt["d"], pt["deliverable_mw"], pt["pooled_coverage"],
                         pt["per_start_coverage_worst"], pt["frac_starts_meeting_target"]))
    print("\ndone in", out["runtime_sec"], "s -> e4e5e6.json")


if __name__ == "__main__":
    main()
