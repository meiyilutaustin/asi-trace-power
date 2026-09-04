#!/usr/bin/env python
"""R4 (reviewer-requested robustness check): exact weighted
covariance decomposition of the cluster -> fleet synchrony ratio, on stable
capacity plateaus, with a PAIRED interval on the raw -> capacity-normalised
change.

An earlier version of the scaling analysis attributed the fleet-level covariance to synchronised capacity expansion.  A reviewer objected that the capacity-normalised
CI is 0.55-1.46 and that the unweighted mean correlation is only 0.22, so the
causal wording outruns the evidence.  This script replaces the ratio talk with
the exact identity and a paired test.

Identity (residual r_i, fleet R = sum_i r_i):
    Var(R) = sum_i var_i + sum_{i != j} cov_ij
    sync^2 = Var(R) / sum_i var_i = 1 + X,   X = excess covariance ratio
    X = rho_w * (S1^2 - S2) / S2,  S1 = sum_i sd_i, S2 = sum_i var_i
    rho_w = sum_{i != j} cov_ij / sum_{i != j} sd_i sd_j   (variance-weighted
            mean correlation; equals the unweighted mean only for equal sd)
Plateaus: segments of >= MIN_DAYS days over which the 168 h rolling mean of
fleet power stays within +-PLATEAU_TOL of the segment mean, i.e. windows with
no capacity step.  If synchronised expansion explains the raw synchrony, X
should collapse within plateaus even WITHOUT capacity normalisation.
Paired interval: week-block bootstrap, same blocks for raw and normalised, so
the CI is on the CHANGE rather than on two independent quantities.
Outputs: summary.json, f1_plateaus.png, f2_excess_covariance.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
ROLL = 169
TREND = 168
MIN_DAYS = 14
PLATEAU_TOL = 0.05
NBOOT = 500
BLOCK = 168
MIN_MEAN_MW = 0.01
MIN_HOURS = 1500


CAP_FLOOR_FRAC = 0.2      # a cluster's local capacity must exceed this share of its median


def resid(wide, kind):
    """Residual matrix (t x clusters), all in MW.

    capnorm: express each cluster's residual at a COMMON capacity level,
        r_i(t) = (w_i - trend_i) * mean_i / cap_i(t),
    which removes the effect of the cluster growing without dividing the level
    itself.  Hours where cap_i falls below CAP_FLOOR_FRAC of the cluster's
    median cap are masked: four clusters idle to ~0 MW for stretches, and the
    unguarded ratio w/cap reaches 168x there, which previously let those four
    clusters carry 97% of the normalised variance (an earlier unguarded version produced that artefact).
    """
    tr = wide.rolling(ROLL, center=True, min_periods=24).mean()
    r = wide - tr
    if kind == "capnorm":
        cap = wide.rolling(TREND, min_periods=24).mean().bfill()
        floor = np.maximum(cap.median() * CAP_FLOOR_FRAC, 0.05)
        valid = cap.ge(floor, axis=1)
        r = (r * (wide.mean() / cap.where(valid))).where(valid)
    if kind in ("deseason", "capnorm"):
        t = np.arange(len(r))
        key = pd.Series((t % 24) + 24 * ((t // 24) % 7), index=r.index)
        r = r - r.groupby(key).transform("mean")
    return r


def decompose(R):
    """R: DataFrame t x clusters of residuals. Returns the exact identity."""
    v = R.var(ddof=0)
    S2 = float(v.sum())
    S1 = float(np.sqrt(v).sum())
    var_sum = float(R.sum(axis=1).var(ddof=0))
    X = var_sum / S2 - 1.0
    denom = S1 ** 2 - S2
    rho_w = X * S2 / denom if denom > 0 else np.nan
    corr = R.corr()
    iu = np.triu_indices(len(corr), 1)
    return {"sync_ratio": float(np.sqrt(var_sum / S2)), "excess_X": float(X), "rho_weighted": float(rho_w),
            "rho_unweighted_mean": float(np.nanmean(corr.values[iu])),
            "hhi_of_variance": float((v / v.sum()).pow(2).sum()), "n_clusters": int(R.shape[1]),
            "var_sum_mw2": var_sum, "sum_var_mw2": S2}


def plateaus(fleet, good):
    """Segments with no capacity step: 168 h rolling mean within +-tol."""
    s = pd.Series(np.where(good, fleet, np.nan)).rolling(TREND, min_periods=24).mean()
    segs = []
    i = 0
    n = len(s)
    while i < n:
        if not np.isfinite(s.iloc[i]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(s.iloc[j]):
            w = s.iloc[i:j + 1]
            if (w.max() - w.min()) / w.mean() > PLATEAU_TOL:
                break
            j += 1
        if (j - i) >= MIN_DAYS * 24:
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def main():
    cl = pd.read_parquet(_find(RERUN, "a5", "envelope_cluster_hourly.parquet"))
    cl["total"] = cl[["floor", "shift", "curtail", "standby"]].sum(axis=1)
    wide = cl.pivot(index="t", columns="cluster_id", values="total")
    good = cl.groupby("t")["good"].first().to_numpy().astype(bool)
    wide[~good] = np.nan
    keep = [c for c in wide.columns if wide[c].mean() >= MIN_MEAN_MW and wide[c].notna().sum() >= MIN_HOURS]
    wide = wide[keep]
    fleet = wide.sum(axis=1).to_numpy()
    out = {"n_clusters": len(keep), "hours": int(good.sum()),
           "fleet_mean_mw": float(np.nanmean(np.where(good, fleet, np.nan)))}

    segs = plateaus(fleet, good)
    out["plateaus"] = [{"start_day": int(a / 24), "end_day": int(b / 24), "hours": int(b - a),
                        "mean_mw": float(np.nanmean(np.where(good[a:b], fleet[a:b], np.nan)))} for a, b in segs]

    # common cluster set: capacity normalisation is only defined for clusters that
    # stay above the capacity floor; use the SAME set for all three residual types
    cap_all = wide.rolling(TREND, min_periods=24).mean().bfill()
    floor_all = np.maximum(cap_all.median() * CAP_FLOOR_FRAC, 0.05)
    valid_frac = cap_all.ge(floor_all, axis=1).loc[good].mean()
    common = [c for c in wide.columns if valid_frac[c] >= 0.8]
    out["capnorm_cluster_set"] = {"kept": [str(c)[:8] for c in common],
                                  "dropped": {str(c)[:8]: float(valid_frac[c]) for c in wide.columns if c not in common},
                                  "rule": f"cluster's 168 h mean stays above max(20% of its median, 0.05 MW) for >= 80% of hours",
                                  "dropped_power_share_pct": float(wide[[c for c in wide.columns if c not in common]].mean().sum() / wide.mean().sum() * 100)}
    wide_c = wide[common]
    R_all = {"raw": resid(wide, "raw")}
    R = {k: resid(wide_c, k) for k in ["raw", "deseason", "capnorm"]}
    ok_rows = {k: good & v.notna().all(axis=1).to_numpy() for k, v in R.items()}
    out["masked_hours"] = {k: {"kept": int(m.sum()), "dropped_vs_good": int(good.sum() - m.sum())} for k, m in ok_rows.items()}
    out["full_window"] = {k: decompose(v.loc[ok_rows[k]]) for k, v in R.items()}
    out["full_window_raw_all_clusters"] = decompose(R_all["raw"].loc[good])
    out["plateau_decomposition"] = []
    for a, b in segs:
        rec = {"start_day": int(a / 24), "end_day": int(b / 24)}
        for k, v in R.items():
            sub = v.iloc[a:b]
            sub = sub.loc[ok_rows[k][a:b]]
            if len(sub) > 24 * 7:
                rec[k] = decompose(sub)
        out["plateau_decomposition"].append(rec)
    if out["plateau_decomposition"]:
        for k in ["raw", "deseason", "capnorm"]:
            vals = [p[k]["sync_ratio"] for p in out["plateau_decomposition"] if k in p]
            xs = [p[k]["excess_X"] for p in out["plateau_decomposition"] if k in p]
            rw = [p[k]["rho_weighted"] for p in out["plateau_decomposition"] if k in p]
            out.setdefault("plateau_pooled", {})[k] = {
                "n_plateaus": len(vals), "sync_ratio_mean": float(np.mean(vals)), "sync_ratio_min": float(np.min(vals)),
                "sync_ratio_max": float(np.max(vals)), "excess_X_mean": float(np.mean(xs)),
                "rho_weighted_mean": float(np.mean(rw))}

    # ---- paired block bootstrap on the CHANGE ------------------------------------
    idx = np.arange(len(wide))
    ok = good
    nblk = int(np.ceil(len(idx) / BLOCK))
    rng = np.random.default_rng(4)
    pairs = {"raw_minus_capnorm": [], "raw_minus_deseason": [], "deseason_minus_capnorm": [],
             "raw": [], "deseason": [], "capnorm": []}
    for _ in range(NBOOT):
        starts = rng.integers(0, len(idx) - BLOCK + 1, nblk)
        sel = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:len(idx)]
        if len(sel) < 500:
            continue
        d = {}
        for k in R:
            s_k = sel[ok_rows[k][sel]]
            if len(s_k) < 500:
                d = {}
                break
            d[k] = decompose(R[k].iloc[s_k])
        if not d:
            continue
        for k in ["raw", "deseason", "capnorm"]:
            pairs[k].append(d[k]["sync_ratio"])
        pairs["raw_minus_capnorm"].append(d["raw"]["sync_ratio"] - d["capnorm"]["sync_ratio"])
        pairs["raw_minus_deseason"].append(d["raw"]["sync_ratio"] - d["deseason"]["sync_ratio"])
        pairs["deseason_minus_capnorm"].append(d["deseason"]["sync_ratio"] - d["capnorm"]["sync_ratio"])
    band = lambda v: {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                      "p95": float(np.percentile(v, 95)), "excludes_zero": bool(np.percentile(v, 5) > 0 or np.percentile(v, 95) < 0)}
    out["paired_bootstrap"] = {k: band(v) for k, v in pairs.items() if v}
    fw = out["full_window"]
    out["attribution"] = {
        "raw_excess_X": fw["raw"]["excess_X"],
        "share_of_excess_removed_by_deseasoning_pct": (fw["raw"]["excess_X"] - fw["deseason"]["excess_X"]) / fw["raw"]["excess_X"] * 100,
        "share_of_excess_removed_by_capacity_normalisation_pct": (fw["raw"]["excess_X"] - fw["capnorm"]["excess_X"]) / fw["raw"]["excess_X"] * 100,
        "residual_excess_after_capnorm": fw["capnorm"]["excess_X"],
        "note": ("excess X = sync^2 - 1 is the exact covariance term; percentages are on the VARIANCE-excess scale, "
                 "which is the scale on which the decomposition is additive")}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps({"full_window": out["full_window"], "attribution": out["attribution"],
                      "paired": out["paired_bootstrap"], "plateau_pooled": out.get("plateau_pooled")}, indent=1))

    # ---- figures -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(fleet)) / 24
    ax.plot(t, np.where(good, fleet, np.nan), lw=0.4, color="k")
    for p in out["plateaus"]:
        ax.axvspan(p["start_day"], p["end_day"], color="tab:green", alpha=0.15)
    ax.set(xlabel="day", ylabel="fleet GPU-side MW", title=f"stable capacity plateaus (±{PLATEAU_TOL:.0%} of the 168 h mean, ≥{MIN_DAYS} days)")
    fig.savefig(os.path.join(OUT, "f1_plateaus.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ks = ["raw", "deseason", "capnorm"]
    lbl = {"raw": "raw residual", "deseason": "− hour×weekday profile", "capnorm": "− capacity trend"}
    axes[0].bar(range(3), [fw[k]["sync_ratio"] for k in ks], color=["tab:red", "tab:orange", "tab:green"])
    for i, k in enumerate(ks):
        b = out["paired_bootstrap"][k]
        axes[0].errorbar(i, fw[k]["sync_ratio"], yerr=[[fw[k]["sync_ratio"] - b["p5"]], [b["p95"] - fw[k]["sync_ratio"]]], color="k", capsize=3)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xticks(range(3), [lbl[k] for k in ks], fontsize=8)
    axes[0].set(ylabel="cluster→fleet synchrony ratio", title="ratio (with block-bootstrap bands)")
    pp = out.get("plateau_pooled", {})
    if pp:
        axes[1].bar(range(3), [pp[k]["excess_X_mean"] for k in ks], color=["tab:red", "tab:orange", "tab:green"], alpha=0.6, label="plateau mean")
    axes[1].bar(range(3), [fw[k]["excess_X"] for k in ks], 0.4, color=["darkred", "darkorange", "darkgreen"], label="full window")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(range(3), [lbl[k] for k in ks], fontsize=8)
    axes[1].set(ylabel="excess covariance X = sync² − 1", title="exact covariance excess")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "f2_excess_covariance.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
