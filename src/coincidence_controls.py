#!/usr/bin/env python3
"""ChatGPT round-3 review, Q2 (coincidence rescue): E1 counterfactual-fidelity
table, E2 single-cluster negative control, E3 synthetic-independent surrogate.

Re-analysis only on the canonical 13-cluster / 4,439 h idle-retained LP-active
substrate (same as runs/2026-09-15_flexibility-decomposition). Reuses the
counterfactual transform (a6 null_indep_v2) exactly as shipped in
src/flex_decomposition.kval_decorrelated: per cluster detrend (168 h centred),
circularly shift the residual, re-add its own trend, sum across clusters, then
max(0,.), then P(.,T,alpha) = (1-alpha) quantile of the T-hour running minimum.

Question: is that transform removing ONLY cross-cluster synchronicity, or is it
also distorting the marginal / low tail (which would make the "coincidence
overstatement" an artifact)?
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)
import flex_decomposition as fx  # noqa: E402

AGG = os.path.abspath(os.path.join(HERE, "..", "..",
      "runs/2026-09-02_a1-load-reconstruction/pod_hourly_agg.parquet"))
CFG = os.path.abspath(os.path.join(HERE, "..", "..", "configs/power_curves.yaml"))
HORIZONS = (1, 4, 24)
ALPHA = 0.95
SEED = 20260918


def acf(x, lag):
    x = np.asarray(x, float)
    v = ~np.isnan(x)
    x = x[v]
    if len(x) <= lag:
        return np.nan
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def counterfactual_aggregate(C, rng, keep_negative=False):
    """One draw of the independent-summed aggregate. Returns (sum_before_trunc)."""
    tr = fx.trend_of(C)
    res = C - tr
    nt = C.shape[1]
    s = np.zeros(nt)
    for i in range(C.shape[0]):
        s += np.roll(res[i], rng.integers(0, nt)) + tr[i]
    return s


# --------------------------------------------------------------------------- E1
def e1_fidelity(C, A, good):
    """Per-cluster and aggregate BEFORE vs AFTER the transform, plus the
    max(0,.) truncation frequency and share of energy it removes."""
    rng = np.random.default_rng(SEED)
    names = None
    tr = fx.trend_of(C)
    res = C - tr

    # Per-cluster: circular shift preserves each cluster's own marginal exactly
    # (it only relabels time), so BEFORE vs AFTER on any marginal statistic must
    # be ~0. We verify this explicitly (kills "the transform distorts marginals").
    per_cluster = []
    n_draw = 50
    for i in range(C.shape[0]):
        orig = C[i]
        v = ~np.isnan(orig)
        # AFTER contributions averaged over draws (before the aggregate sum/trunc)
        after_mean = []
        after_std = []
        after_neg_frac = []
        after_k = {h: [] for h in HORIZONS}
        for _ in range(n_draw):
            shifted = np.roll(res[i], rng.integers(0, C.shape[1])) + tr[i]
            after_mean.append(np.nanmean(shifted))
            after_std.append(np.nanstd(shifted[v]))
            after_neg_frac.append(float(np.mean(shifted[v] < 0)))
            for h in HORIZONS:
                after_k[h].append(fx.kval(shifted, h, ALPHA))
        per_cluster.append(dict(
            cluster=i,
            mean_before=float(np.nanmean(orig)),
            mean_after=float(np.mean(after_mean)),
            std_before=float(np.nanstd(orig[v])),
            std_after=float(np.mean(after_std)),
            acf1_before=acf(orig, 1),
            acf24_before=acf(orig, 24),
            k95_1h_before=fx.kval(orig, 1, ALPHA),
            k95_1h_after=float(np.mean(after_k[1])),
            k95_24h_before=fx.kval(orig, 24, ALPHA),
            k95_24h_after=float(np.mean(after_k[24])),
            transformed_negative_hour_frac=float(np.mean(after_neg_frac)),
        ))

    # Aggregate: actual A vs counterfactual aggregate (averaged over draws), and
    # the truncation diagnostics on the summed counterfactual.
    n_agg = 200
    agg_mean, agg_std, agg_acf1, agg_acf24 = [], [], [], []
    agg_k = {h: [] for h in HORIZONS}
    trunc_hour_frac, trunc_energy_share = [], []
    for _ in range(n_agg):
        s = counterfactual_aggregate(C, rng)
        v = ~np.isnan(s)
        neg = (s < 0) & v
        trunc_hour_frac.append(float(neg.sum() / v.sum()))
        pos_energy = np.nansum(np.maximum(s, 0.0))
        removed = -np.nansum(np.minimum(s, 0.0))  # magnitude of clipped negatives
        trunc_energy_share.append(float(removed / (pos_energy + removed)) if pos_energy + removed > 0 else 0.0)
        st = np.maximum(s, 0.0)
        st[~v] = np.nan
        agg_mean.append(np.nanmean(st))
        agg_std.append(np.nanstd(st[v]))
        agg_acf1.append(acf(st, 1)); agg_acf24.append(acf(st, 24))
        for h in HORIZONS:
            agg_k[h].append(fx.kval(st, h, ALPHA))

    aggregate = dict(
        actual_mean=float(np.nanmean(A)),
        cf_mean=float(np.mean(agg_mean)),
        actual_std=float(np.nanstd(A[~np.isnan(A)])),
        cf_std=float(np.mean(agg_std)),
        actual_acf1=acf(A, 1), cf_acf1=float(np.mean(agg_acf1)),
        actual_acf24=acf(A, 24), cf_acf24=float(np.mean(agg_acf24)),
        actual_k95={h: fx.kval(A, h, ALPHA) for h in HORIZONS},
        cf_k95={h: float(np.mean(agg_k[h])) for h in HORIZONS},
        a_coinc={h: fx.kval(A, h, ALPHA) / float(np.mean(agg_k[h])) for h in HORIZONS},
        truncation_hour_frac_mean=float(np.mean(trunc_hour_frac)),
        truncation_hour_frac_max=float(np.max(trunc_hour_frac)),
        truncation_energy_share_mean=float(np.mean(trunc_energy_share)),
        truncation_energy_share_max=float(np.max(trunc_energy_share)),
        n_draws=n_agg,
    )
    return dict(per_cluster=per_cluster, aggregate=aggregate)


# --------------------------------------------------------------------------- E2
def e2_single_cluster(C, good):
    """Run the full detrend->shift->restore->truncate->firm-power pipeline on ONE
    cluster at a time (no cross-cluster synchronicity to remove). a_coinc should
    be ~1 at every horizon; a systematic move = the procedure manufactures a
    coincidence penalty."""
    rng = np.random.default_rng(SEED + 1)
    n_shift = 200
    rows = []
    for i in range(C.shape[0]):
        Ci = C[i:i + 1]                    # 1 x nt
        Ai = C[i]
        row = dict(cluster=i)
        for h in HORIZONS:
            P2 = fx.kval_decorrelated(Ci, h, ALPHA, rng, n_shift)   # independent (=itself)
            P3 = fx.kval(Ai, h, ALPHA)                              # honest
            row[f"a_coinc_{h}h"] = float(P3 / P2) if P2 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {f"a_coinc_{h}h": dict(
        mean=float(df[f"a_coinc_{h}h"].mean()),
        median=float(df[f"a_coinc_{h}h"].median()),
        min=float(df[f"a_coinc_{h}h"].min()),
        max=float(df[f"a_coinc_{h}h"].max()),
    ) for h in HORIZONS}
    return dict(per_cluster=rows, summary=summary,
                note="a_coinc = honest / independent for a single cluster; deviation from 1 "
                     "is pure procedure noise (shift + running-min reordering + truncation), "
                     "not real coincidence.")


# --------------------------------------------------------------------------- E3
def phase_randomize(res_row, rng):
    """Amplitude-adjusted phase-randomized surrogate: preserve the power spectrum
    (hence variance + autocorrelation) but randomize phases -> independent across
    clusters when each is drawn with its own random phases."""
    x = np.asarray(res_row, float)
    v = ~np.isnan(x)
    y = x.copy()
    y[~v] = np.nanmean(x)              # fill gaps for the FFT
    n = len(y)
    F = np.fft.rfft(y)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, len(F)))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    surrogate = np.fft.irfft(np.abs(F) * phases, n=n)
    surrogate[~v] = np.nan
    return surrogate


def e3_synthetic_independent(C, good, n_real=200):
    """Build KNOWN-independent synthetic cluster panels (phase-randomized
    surrogates: each cluster keeps its own trend, variance and autocorrelation,
    but cross-cluster dependence is destroyed), then run the counterfactual.
    Expected a_coinc ~ 1 (within CI); a systematic offset = a bias floor."""
    rng = np.random.default_rng(SEED + 2)
    tr = fx.trend_of(C)
    res = C - tr
    n_shift = 50
    samples = {h: [] for h in HORIZONS}
    for _ in range(n_real):
        # independent synthetic panel: own trend + phase-randomized own residual
        Csyn = np.vstack([tr[i] + phase_randomize(res[i], rng) for i in range(C.shape[0])])
        Csyn[:, ~good] = np.nan
        Asyn = np.nansum(Csyn, axis=0)
        Asyn[~good] = np.nan
        for h in HORIZONS:
            P2 = fx.kval_decorrelated(Csyn, h, ALPHA, rng, n_shift)
            P3 = fx.kval(Asyn, h, ALPHA)
            samples[h].append(P3 / P2 if P2 else np.nan)
    summary = {}
    for h in HORIZONS:
        arr = np.array(samples[h], float)
        arr = arr[~np.isnan(arr)]
        summary[f"a_coinc_{h}h"] = dict(
            mean=float(arr.mean()), p5=float(np.percentile(arr, 5)),
            p50=float(np.percentile(arr, 50)), p95=float(np.percentile(arr, 95)),
            n=int(len(arr)))
    return dict(summary=summary, n_realizations=n_real,
                note="phase-randomized surrogates preserve each cluster's spectrum "
                     "(variance+autocorr) and trend, destroy cross-cluster dependence. "
                     "a_coinc materially below 1 here would be a bias floor to subtract.")


def main():
    t0 = time.time()
    C, A, Wf, good, meta = fx.build_substrate(AGG, CFG)
    out = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        substrate=dict(agg=AGG, cfg=CFG, sha256_agg=fx.sha256(AGG),
                       n_clusters=meta["n_clusters"], valid_hours=meta["valid_hours"],
                       mean_eligible_mw=meta["mean_eligible_mw"],
                       scope_share=meta["measured_scope_share"]),
        alpha=ALPHA, horizons=list(HORIZONS), seed=SEED,
    )
    print("E1 fidelity ..."); out["E1_fidelity"] = e1_fidelity(C, A, good)
    print("E2 single-cluster control ..."); out["E2_single_cluster"] = e2_single_cluster(C, good)
    print("E3 synthetic-independent ..."); out["E3_synthetic_independent"] = e3_synthetic_independent(C, good)
    out["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "e1e2e3.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("done in", out["runtime_sec"], "s -> e1e2e3.json")

    # console headline
    agg = out["E1_fidelity"]["aggregate"]
    print("\n== E1 aggregate ==")
    print("  actual mean %.3f  cf mean %.3f" % (agg["actual_mean"], agg["cf_mean"]))
    print("  low-tail k95 actual/cf a_coinc:", {k: round(v, 3) for k, v in agg["a_coinc"].items()})
    print("  truncation hour-frac mean %.2e max %.2e" % (agg["truncation_hour_frac_mean"], agg["truncation_hour_frac_max"]))
    print("  truncation energy-share mean %.2e max %.2e" % (agg["truncation_energy_share_mean"], agg["truncation_energy_share_max"]))
    print("== E2 single-cluster a_coinc (should ~1) ==")
    for h in HORIZONS:
        s = out["E2_single_cluster"]["summary"][f"a_coinc_{h}h"]
        print("  %2dh: mean %.3f  [min %.3f, max %.3f]" % (h, s["mean"], s["min"], s["max"]))
    print("== E3 synthetic-independent a_coinc (should ~1) ==")
    for h in HORIZONS:
        s = out["E3_synthetic_independent"]["summary"][f"a_coinc_{h}h"]
        print("  %2dh: mean %.3f  [p5 %.3f, p95 %.3f]" % (h, s["mean"], s["p5"], s["p95"]))


if __name__ == "__main__":
    main()
