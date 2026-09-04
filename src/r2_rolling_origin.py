#!/usr/bin/env python
"""R2 (reviewer-requested robustness check): rolling-origin validation of K.

A first-half/second-half holdout is flattered by growth: the eligible mean rises from 3.24 to 3.82 MW, so a K fitted on the
early half is easy to meet later.  This script does prospective validation:

  for each origin o (weekly step):
      train on the W weeks before o, test on the T weeks after o
      K_abs   = K(alpha, h) fitted on the train window, applied as MW
      K_norm  = (K_train / mean_train) * mean_test   (capacity-normalised: the
                planner re-scales last period's ratio by the current mean)
      coverage = P[ running-min over the test window >= K ]
      shortfall = mean and worst relative depth when the window falls short
  report coverage vs target, and the DERATING factor d such that d*K reaches
  the target coverage in the pooled test windows.

Both units are reported because the two answer different questions: K_abs asks
"does a number fixed last quarter still hold", K_norm asks "does the RATIO hold".
Outputs: summary.json, f1_rolling_coverage.png, f2_derating.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


def _find(base, *parts):
    """Locate a product: flat under base (data/products), a stage subdir (run dir),
    or the public results/ tree next to data/ (summary tables)."""
    import os as _os
    stage_map = {"a1": "a1_load", "a5": "a5_envelope", "a6": "a6_availability"}
    cands = [_os.path.join(base, parts[-1]), _os.path.join(base, *parts)]
    root = _os.path.abspath(_os.path.join(base, _os.pardir, _os.pardir))
    if len(parts) == 2 and parts[0] in stage_map:
        cands.append(_os.path.join(root, "results", stage_map[parts[0]], parts[1]))
    for c in cands:
        if _os.path.exists(c):
            return c
    return cands[-1]

RERUN = sys.argv[1]   # data/products (public layout) or a run directory with stage subdirs
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
HS = [1, 4, 24]
ALPHAS = [0.9, 0.95]
TRAIN_W = [4, 8, 12]      # weeks
TEST_W = 4                # weeks
STEP_W = 1


def rmin(s, h):
    m = sliding_window_view(s, h).min(axis=1) if h > 1 else s
    return m[~np.isnan(m)]


def kval(s, h, a):
    m = rmin(s, h)
    return float(np.quantile(m, 1 - a)) if len(m) else np.nan


def main():
    env = pd.read_parquet(_find(RERUN, "a5", "envelope_hourly.parquet"))
    good = env["good"].to_numpy().astype(bool)
    C = np.where(good, env["curtail_idle_retained"].to_numpy(), np.nan)
    nt = len(C)
    W = 168
    res = {"inputs": {"hours": int(good.sum()), "mean_mw": float(np.nanmean(C)),
                      "test_weeks": TEST_W, "step_weeks": STEP_W}}
    rows = []
    for tw in TRAIN_W:
        o = tw * W
        while o + TEST_W * W <= nt:
            tr = C[o - tw * W:o]
            te = C[o:o + TEST_W * W]
            if np.isfinite(tr).sum() < 0.5 * len(tr) or np.isfinite(te).sum() < 0.5 * len(te):
                o += STEP_W * W
                continue
            m_tr, m_te = np.nanmean(tr), np.nanmean(te)
            for h in HS:
                mt = rmin(te, h)
                if len(mt) < 10:
                    continue
                for a in ALPHAS:
                    k_abs = kval(tr, h, a)
                    k_norm = (k_abs / m_tr) * m_te
                    for tag, k in [("abs", k_abs), ("norm", k_norm)]:
                        cov = float(np.mean(mt >= k))
                        short = mt[mt < k]
                        rows.append({"train_weeks": tw, "origin_day": o // 24, "h": h, "alpha": a, "unit": tag,
                                     "K_mw": k, "coverage": cov, "target": a,
                                     "mean_shortfall_frac": float(np.mean((k - short) / k)) if len(short) else 0.0,
                                     "worst_shortfall_frac": float(np.max((k - short) / k)) if len(short) else 0.0,
                                     "mean_train_mw": float(m_tr), "mean_test_mw": float(m_te),
                                     "growth_ratio": float(m_te / m_tr)})
            o += STEP_W * W
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "rolling_origin_windows.csv"), index=False)

    # pooled coverage and derating
    summ = {}
    for tw in TRAIN_W:
        for h in HS:
            for a in ALPHAS:
                for tag in ["abs", "norm"]:
                    sub = df[(df.train_weeks == tw) & (df.h == h) & (df.alpha == a) & (df.unit == tag)]
                    if not len(sub):
                        continue
                    summ[f"tw{tw}_h{h}_a{a}_{tag}"] = {
                        "n_origins": int(len(sub)), "coverage_mean": float(sub.coverage.mean()),
                        "coverage_p10": float(sub.coverage.quantile(0.1)), "coverage_min": float(sub.coverage.min()),
                        "frac_origins_meeting_target": float((sub.coverage >= a).mean()),
                        "mean_shortfall_frac": float(sub.mean_shortfall_frac.mean()),
                        "worst_shortfall_frac": float(sub.worst_shortfall_frac.max()),
                        "mean_growth_ratio": float(sub.growth_ratio.mean())}
    res["by_setting"] = summ

    # derating: smallest d such that pooled coverage >= target (per h, alpha, unit, tw=8)
    der = {}
    for h in HS:
        for a in ALPHAS:
            for tag in ["abs", "norm"]:
                sub = df[(df.train_weeks == 8) & (df.h == h) & (df.alpha == a) & (df.unit == tag)]
                if not len(sub):
                    continue
                lo, hi = 0.3, 1.5
                for _ in range(40):
                    d = 0.5 * (lo + hi)
                    covs = []
                    o = 8 * W
                    i = 0
                    while o + TEST_W * W <= nt and i < len(sub):
                        tr = C[o - 8 * W:o]; te = C[o:o + TEST_W * W]
                        if np.isfinite(tr).sum() >= 0.5 * len(tr) and np.isfinite(te).sum() >= 0.5 * len(te):
                            mt = rmin(te, h)
                            if len(mt) >= 10:
                                k = kval(tr, h, a)
                                if tag == "norm":
                                    k = k / np.nanmean(tr) * np.nanmean(te)
                                covs.append(np.mean(mt >= d * k))
                                i += 1
                        o += STEP_W * W
                    if np.mean(covs) >= a:
                        lo = d
                    else:
                        hi = d
                    if tag and abs(hi - lo) < 1e-3:
                        break
                der[f"h{h}_a{a}_{tag}"] = {"derating_factor_for_target": float(lo),
                                           "derated_K_mw": float(lo * kval(C, h, a))}
    res["derating_trainweeks8"] = der
    res["interpretation"] = ("coverage close to or above the target with a derating factor near 1 means K generalises "
                             "prospectively; a factor well below 1 means the in-sample K is optimistic and the paper "
                             "must call K a retrospective lower-tail statistic.")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["derating_trainweeks8"]}, indent=1))
    for k in ["tw8_h4_a0.95_abs", "tw8_h4_a0.95_norm", "tw8_h24_a0.95_abs", "tw8_h1_a0.95_abs"]:
        if k in summ:
            print(k, {kk: round(v, 3) for kk, v in summ[k].items()})

    # ---- figures ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, tag, ttl in [(axes[0], "abs", "K fixed in MW"), (axes[1], "norm", "K re-scaled by the current mean")]:
        for h, c in zip(HS, ["tab:blue", "tab:green", "tab:red"]):
            sub = df[(df.train_weeks == 8) & (df.h == h) & (df.alpha == 0.95) & (df.unit == tag)].sort_values("origin_day")
            ax.plot(sub.origin_day, sub.coverage, "o-", ms=3, lw=0.9, color=c, label=f"h = {h} h")
        ax.axhline(0.95, color="k", ls=":", lw=1, label="target 0.95")
        ax.set(xlabel="origin (day)", title=ttl, ylim=(0, 1.05))
    axes[0].set_ylabel("coverage in the next 4 weeks")
    axes[0].legend(fontsize=8)
    fig.suptitle("rolling-origin validation of K(0.95, h), 8-week training windows")
    fig.savefig(os.path.join(OUT, "f1_rolling_coverage.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(HS))
    for j, (a, tag) in enumerate([(0.95, "abs"), (0.95, "norm"), (0.9, "abs")]):
        vals = [der.get(f"h{h}_a{a}_{tag}", {}).get("derating_factor_for_target", np.nan) for h in HS]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, label=f"α = {a}, {'MW' if tag == 'abs' else 'ratio'}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(x, [f"{h} h" for h in HS])
    ax.set(ylabel="derating factor needed to hit the target", title="how much K must be derated to generalise forward")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "f2_derating.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
