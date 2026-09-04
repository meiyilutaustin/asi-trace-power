#!/usr/bin/env python
"""A6 (v2, revision 2): duration-reliability-aggregation surface of
eligible curtailment.

    K_{alpha,h}(S) = (1-alpha) quantile of the h-hour running minimum of the
                     portfolio's eligible-curtailable power C_S(t)

C uses the shared power model (online floor) and the curtail boundary from the
config (default idle_retained: the reclaimed GPU keeps idling).  Products:
  curtail      LP eligible curtailment only
  curtail+s_h  + HP shiftable layer scaled by the horizon-specific observed
               queueing-delay share s_h (h<=1: s_1; 2-4: s_4; >4: s_24 = 0)  
Baselines (mis-specifications a planner might use):
  const   constant share of load calibrated to the MEAN eligible share
  shuf    hours jointly permuted (marginals + covariance kept, persistence lost)
  indep   v2: each cluster's DETRENDED residual circularly shifted and its own
          168 h trend added back (covariance lost, growth phases kept) ;
          v1 (raw circular shift) also reported for comparison
Uncertainty: B replicates = power-MC draw x block bootstrap (block 168 h; 24 and
336 h sensitivity), with PAIRED baselines inside each replicate .
Robustness : eras (day<105 / >=105 and Standby first day), monthly K,
first-half -> second-half holdout coverage, effective sample counts.
Missing hours are NaN gaps: any window that spans one is dropped .
Electrical boundary columns: workload (GPU side), facility marginal (x1.0),
facility average (x PUE) .
"""
import itertools
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import powermodel as pm  # noqa: E402

DATA = os.environ.get("DATA_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "data"))
AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a6_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(
    os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
HS = [1, 2, 4, 8, 24]
ALPHAS = [0.5, 0.9, 0.95, 0.99]
HEAD = [(1, 0.95), (4, 0.95), (24, 0.95), (4, 0.9), (4, 0.99)]
N_SUB = int(os.environ.get("A6_NSUB", 200))
N_NULL = int(os.environ.get("A6_NNULL", 20))
N_NULL_PAIRED = int(os.environ.get("A6_NNULL_PAIRED", 8))
B = int(os.environ.get("A6_BOOT", 200))
BLOCKS = [24, 168, 336]
BLOCK = 168
MIN_CLUSTER_MW = 0.02
REGIME_DAY = 105
TREND_H = 168


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def tolerance():
    js = pd.read_parquet(os.path.join(DATA, "asi_opensource_job_execution_summary"),
                         columns=["gpu_request", "duration_hours", "schedule_delay_sec",
                                  "job_type_public", "priority_class"])
    js["w"] = js["gpu_request"].fillna(0).clip(lower=0) * js["duration_hours"].fillna(0).clip(lower=0)
    js["dh"] = js["schedule_delay_sec"].fillna(0).clip(lower=0) / 3600
    out = {}
    for (jt, pr), g in js.groupby(["job_type_public", "priority_class"], observed=True):
        tot = g["w"].sum()
        if tot > 0:
            out[(jt, pr)] = {t: float(g.loc[g["dh"] >= t, "w"].sum() / tot) for t in (1, 4, 24)}
    return out


def s_for_h(h):
    return 1 if h <= 1 else (4 if h <= 4 else 24)


def build(parts, layer, s_tau, cl_codes, tpod, ncl, nt, boundary):
    curt = np.zeros((ncl, nt))
    work = np.zeros((ncl, nt))
    m = layer == "curtail"
    np.add.at(curt, (cl_codes[m], tpod[m]), pm.curtail_energy(parts, boundary)[m] / 1e6)
    np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
    shift = {}
    ms = layer == "shift"
    for tau in (1, 4, 24):
        a = np.zeros((ncl, nt))
        np.add.at(a, (cl_codes[ms], tpod[ms]), (parts["total"] * s_tau[tau])[ms] / 1e6)
        shift[tau] = a
    return curt, work, shift


def with_gaps(mat, good):
    out = mat.astype(float).copy()
    out[:, ~good] = np.nan
    return out


def rolling_min(series, h):
    if h == 1:
        m = series
    else:
        m = sliding_window_view(series, h).min(axis=1)   # NaN propagates -> window dropped
    return m[~np.isnan(m)]


def kval(series, h, alpha):
    m = rolling_min(series, h)
    return float(np.quantile(m, 1 - alpha)) if len(m) else np.nan


def eff_counts(series, h, alpha):
    m = rolling_min(series, h)
    n = len(m)
    return {"n_windows": int(n), "n_nonoverlapping": int(n // h),
            "n_tail_windows": int(np.floor((1 - alpha) * n))}


def trend_of(mat):
    """168 h centred rolling mean per cluster (NaN-aware)."""
    return np.vstack([pd.Series(r).rolling(TREND_H, center=True, min_periods=24).mean()
                      .bfill().ffill().to_numpy() for r in mat])


def null_indep_v2(mat, rng, h, a, n=N_NULL):
    tr = trend_of(mat)
    res = mat - tr
    nt = mat.shape[1]
    vals = []
    for _ in range(n):
        s = np.zeros(nt)
        for i in range(mat.shape[0]):
            s += np.roll(res[i], rng.integers(0, nt)) + tr[i]
        vals.append(kval(np.maximum(s, 0.0), h, a))
    return float(np.nanmean(vals))


def null_indep_v1(mat, rng, h, a, n=N_NULL):
    nt = mat.shape[1]
    vals = [kval(sum(np.roll(mat[i], rng.integers(0, nt)) for i in range(mat.shape[0])), h, a)
            for _ in range(n)]
    return float(np.nanmean(vals))


def null_shuf(agg, rng, h, a, n=N_NULL):
    ok = ~np.isnan(agg)
    vals = []
    for _ in range(n):
        s = agg.copy()
        s[ok] = agg[ok][rng.permutation(ok.sum())]
        vals.append(kval(s, h, a))
    return float(np.nanmean(vals))


def const_baseline(agg, w, h, a):
    return float(np.nanmean(agg) / np.nanmean(w) * kval(w, h, a))


def main():
    cfg = yaml.safe_load(open(CFG))
    boundary = cfg.get("curtail_boundary", "idle_retained")
    elig = cfg.get("eligibility", "central")
    pue = cfg["pue"]["base"]
    rng = np.random.default_rng(int(cfg.get("mc_seed", 0)) + 6)
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    cl_codes_all, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
    ncl_all = len(cl_uniq)
    layer = pm.assign_layer(pod, elig)
    tol = tolerance()
    keys = list(zip(pod["job_type_public"].astype(str), pod["priority_class"].astype(str)))
    s_tau = {tau: np.array([tol.get(k, {tau: 0.0})[tau] for k in keys]) for tau in (1, 4, 24)}
    sb = pod[(pod["state_public"] == "Standby") & (pod["gpu_hours"] > 0)]
    standby_day = int(sb["day"].astype(int).min()) if len(sb) else None
    log(f"rows {len(pod)}, hours {nt}, clusters {ncl_all}, boundary {boundary}, eligibility {elig}")

    # ---- observed (base params) ----------------------------------------------
    p0 = pm.draw_params(cfg, rng, specs, base=True)
    parts0 = pm.row_energy(pod, cfg, p0, spec_codes, spec_uniq)
    curt, work, shift = build(parts0, layer, s_tau, cl_codes_all, tpod, ncl_all, nt, boundary)
    good = work.sum(axis=0) > 0
    order = np.argsort(-curt.mean(axis=1))
    keep = [i for i in order if curt[i].mean() >= MIN_CLUSTER_MW]
    excluded = [{"cluster": str(cl_uniq[i])[:8], "mean_curtail_mw": float(curt[i].mean()),
                 "mean_workload_mw": float(work[i].mean())} for i in order if i not in keep]
    N = len(keep)
    names = [str(cl_uniq[i])[:8] for i in keep]
    curt_g = with_gaps(curt[keep], good)
    work_g = with_gaps(work[keep], good)
    shift_g = {tau: with_gaps(shift[tau][keep], good) for tau in shift}
    fleet_c = np.nansum(curt_g, axis=0); fleet_c[~good] = np.nan
    fleet_w = np.nansum(work_g, axis=0); fleet_w[~good] = np.nan
    fleet_s = {tau: np.where(good, np.nansum(shift_g[tau], axis=0), np.nan) for tau in shift}
    log(f"clusters with eligible curtailment >= {MIN_CLUSTER_MW} MW: {N} (excluded {len(excluded)})")

    rows = []
    for h in HS:
        for a in ALPHAS:
            base_row = dict(scope="fleet", n=N, h=h, alpha=a,
                            mean=float(np.nanmean(fleet_c)),
                            k_obs=kval(fleet_c, h, a),
                            k_const=const_baseline(fleet_c, fleet_w, h, a),
                            k_shuf=null_shuf(fleet_c, rng, h, a),
                            k_indep_v2=null_indep_v2(curt_g, rng, h, a),
                            k_indep_v1=null_indep_v1(curt_g, rng, h, a),
                            **eff_counts(fleet_c, h, a))
            prod = fleet_c + fleet_s[s_for_h(h)]
            base_row["k_obs_curtail_plus_shift_h"] = kval(prod, h, a)
            base_row["mean_curtail_plus_shift_h"] = float(np.nanmean(prod))
            rows.append(base_row)
    # portfolios: enumerate when feasible 
    for n in range(1, N + 1):
        if comb(N, n) <= 3003:
            combos = list(itertools.combinations(range(N), n))
        else:
            combos = [tuple(sorted(rng.choice(N, n, replace=False))) for _ in range(N_SUB)]
        for h in HS:
            for a in ALPHAS:
                ko, kc, mu = [], [], []
                for cmb in combos:
                    idx = list(cmb)
                    agg = np.where(good, np.nansum(curt_g[idx], axis=0), np.nan)
                    w = np.where(good, np.nansum(work_g[idx], axis=0), np.nan)
                    ko.append(kval(agg, h, a)); kc.append(const_baseline(agg, w, h, a)); mu.append(np.nanmean(agg))
                rec = dict(scope="random", n=n, h=h, alpha=a, k_obs=float(np.mean(ko)),
                           k_const=float(np.mean(kc)), mean=float(np.mean(mu)),
                           n_portfolios=len(combos), k_shuf=np.nan, k_indep_v1=np.nan, k_indep_v2=np.nan)
                if n > 1 and (h, a) == (4, 0.95):
                    sub = combos if len(combos) <= 40 else [combos[i] for i in rng.choice(len(combos), 40, replace=False)]
                    rec["k_indep_v2"] = float(np.mean([null_indep_v2(curt_g[list(c)], rng, h, a, n=6) for c in sub]))
                    rec["k_indep_v1"] = float(np.mean([null_indep_v1(curt_g[list(c)], rng, h, a, n=6) for c in sub]))
                rows.append(rec)
        idx = list(range(n))
        agg = np.where(good, np.nansum(curt_g[idx], axis=0), np.nan)
        w = np.where(good, np.nansum(work_g[idx], axis=0), np.nan)
        for h in HS:
            for a in ALPHAS:
                rows.append(dict(scope="nested", n=n, h=h, alpha=a, k_obs=kval(agg, h, a),
                                 k_const=const_baseline(agg, w, h, a), mean=float(np.nanmean(agg)),
                                 k_shuf=np.nan, k_indep_v1=np.nan,
                                 k_indep_v2=null_indep_v2(curt_g[idx], rng, h, a, n=6) if n > 1 else kval(agg, h, a)))
    # eras 
    day = np.arange(nt) // 24
    eras = {"pre105": day < REGIME_DAY, "post105": day >= REGIME_DAY,
            "first_half": np.arange(nt) < nt // 2, "second_half": np.arange(nt) >= nt // 2}
    if standby_day is not None:
        eras[f"pre_standby{standby_day}"] = day < standby_day
        eras[f"post_standby{standby_day}"] = day >= standby_day
    for tag, m in eras.items():
        agg = np.where(m, fleet_c, np.nan)
        w = np.where(m, fleet_w, np.nan)
        for h in HS:
            for a in ALPHAS:
                rows.append(dict(scope=tag, n=N, h=h, alpha=a, k_obs=kval(agg, h, a),
                                 k_const=const_baseline(agg, w, h, a), mean=float(np.nanmean(agg)),
                                 k_shuf=np.nan, k_indep_v1=np.nan, k_indep_v2=np.nan, **eff_counts(agg, h, a)))
    surf = pd.DataFrame(rows)
    for c in ["obs", "const", "shuf", "indep_v1", "indep_v2"]:
        surf[f"firm_{c}"] = surf[f"k_{c}"] / surf["mean"]
    # monthly
    monthly = []
    for m0 in range(0, int(day.max()) + 1, 30):
        m = (day >= m0) & (day < m0 + 30)
        agg = np.where(m, fleet_c, np.nan)
        monthly.append({"start_day": m0, "mean_mw": float(np.nanmean(agg)),
                        "k_95_4h": kval(agg, 4, 0.95), "firm_95_4h": kval(agg, 4, 0.95) / float(np.nanmean(agg))})
    # holdout coverage: K from first half, coverage in second half
    holdout = {}
    for h in HS:
        for a in ALPHAS:
            k1 = kval(np.where(eras["first_half"], fleet_c, np.nan), h, a)
            m2 = rolling_min(np.where(eras["second_half"], fleet_c, np.nan), h)
            holdout[f"h{h}_a{a}"] = {"k_first_half": k1, "coverage_second_half": float(np.mean(m2 >= k1)),
                                     "target": a, "n_windows_second_half": int(len(m2))}
    log("observed surface done")

    # ---- uncertainty: power MC x block bootstrap, PAIRED baselines ----------
    boot = {bl: {(h, a): np.zeros(B) for h in HS for a in ALPHAS} for bl in BLOCKS}
    paired = {(h, a): {k: np.zeros(B) for k in ["obs", "const", "shuf", "indep_v2"]} for (h, a) in HEAD}
    boot_firm_n = np.zeros((B, N))
    for b in range(B):
        p = pm.draw_params(cfg, rng, specs, base=(b == 0))
        parts = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)
        c_b, w_b, _ = build(parts, layer, s_tau, cl_codes_all, tpod, ncl_all, nt, boundary)
        c_b = with_gaps(c_b[keep], good); w_b = with_gaps(w_b[keep], good)
        fc = np.where(good, np.nansum(c_b, axis=0), np.nan)
        fw = np.where(good, np.nansum(w_b, axis=0), np.nan)
        for bl in BLOCKS:
            if b == 0:
                idx = np.arange(nt)
            else:
                nblk = int(np.ceil(nt / bl))
                starts = rng.integers(0, nt - bl + 1, nblk)
                idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:nt]
            fcb = fc[idx].copy()
            # block boundaries are NaN gaps so windows never straddle a splice
            if b > 0 and bl < nt:
                fcb[bl - 1::bl] = np.nan
            for h in HS:
                for a in ALPHAS:
                    boot[bl][(h, a)][b] = kval(fcb, h, a)
        # paired baselines on the 168 h replicate
        if b == 0:
            idx = np.arange(nt)
        else:
            nblk = int(np.ceil(nt / BLOCK))
            starts = rng.integers(0, nt - BLOCK + 1, nblk)
            idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:nt]
        cb = c_b[:, idx]; wb = w_b[:, idx]
        fcb = np.nansum(cb, axis=0); fcb[np.isnan(cb).all(axis=0)] = np.nan
        fwb = np.nansum(wb, axis=0); fwb[np.isnan(wb).all(axis=0)] = np.nan
        for (h, a) in HEAD:
            paired[(h, a)]["obs"][b] = kval(fcb, h, a)
            paired[(h, a)]["const"][b] = const_baseline(fcb, fwb, h, a)
            paired[(h, a)]["shuf"][b] = null_shuf(fcb, rng, h, a, n=N_NULL_PAIRED)
            paired[(h, a)]["indep_v2"][b] = null_indep_v2(np.nan_to_num(cb), rng, h, a, n=N_NULL_PAIRED)
        for n in range(1, N + 1):
            agg = np.nansum(cb[:n], axis=0); agg[np.isnan(cb[:n]).all(axis=0)] = np.nan
            boot_firm_n[b, n - 1] = kval(agg, 4, 0.95) / np.nanmean(agg)
        if (b + 1) % 25 == 0:
            log(f"bootstrap {b+1}/{B}")

    band = lambda v: {"p5": float(np.nanpercentile(v, 5)), "p50": float(np.nanpercentile(v, 50)),
                      "p95": float(np.nanpercentile(v, 95)), "base": float(v[0])}
    fl = surf[surf.scope == "fleet"]
    pick = lambda df, h, a: df[(df.h == h) & (df.alpha == a)].iloc[0]
    head = {}
    for (h, a) in HEAD:
        r = pick(fl, h, a)
        pr = paired[(h, a)]
        head[f"h{h}_a{a}"] = {
            "k_obs_mw": {"workload": r.k_obs, "facility_marginal": r.k_obs, "facility_avg_pue": r.k_obs * pue},
            "firm_obs": r.firm_obs,
            "k_obs_curtail_plus_shift_h": r.k_obs_curtail_plus_shift_h,
            "k_const_mw": r.k_const, "k_shuf_mw": r.k_shuf,
            "k_indep_v2_mw": r.k_indep_v2, "k_indep_v1_mw": r.k_indep_v1,
            "const_over_obs": r.k_const / r.k_obs, "indep_v2_over_obs": r.k_indep_v2 / r.k_obs,
            "indep_v1_over_obs": r.k_indep_v1 / r.k_obs, "shuf_over_obs": r.k_shuf / r.k_obs,
            "const_over_obs_band": band(pr["const"] / pr["obs"]),
            "indep_v2_over_obs_band": band(pr["indep_v2"] / pr["obs"]),
            "shuf_over_obs_band": band(pr["shuf"] / pr["obs"]),
            "boot_band_mw_168h": band(boot[168][(h, a)]),
            "boot_band_mw_by_block": {str(bl): band(boot[bl][(h, a)]) for bl in BLOCKS},
            "effective_counts": {k: int(r[k]) for k in ["n_windows", "n_nonoverlapping", "n_tail_windows"]}}
    rnd = surf[surf.scope == "random"]
    port = {int(n): {"firm_obs": float(pick(rnd[rnd.n == n], 4, 0.95).firm_obs),
                     "firm_indep_v2": float(pick(rnd[rnd.n == n], 4, 0.95).firm_indep_v2) if n > 1 else None,
                     "firm_indep_v1": float(pick(rnd[rnd.n == n], 4, 0.95).firm_indep_v1) if n > 1 else None,
                     "mean_mw": float(pick(rnd[rnd.n == n], 4, 0.95)["mean"]),
                     "n_portfolios": int(pick(rnd[rnd.n == n], 4, 0.95).n_portfolios)}
            for n in range(1, N + 1)}
    summary = {
        "settings": {"boundary": boundary, "eligibility": elig, "pue_avg": pue, "B": B,
                     "blocks": BLOCKS, "n_sub": N_SUB, "n_null": N_NULL, "min_cluster_mw": MIN_CLUSTER_MW,
                     "regime_day": REGIME_DAY, "standby_first_day": standby_day,
                     "s_h_rule": "h<=1: s_1; 2<=h<=4: s_4; h>4: s_24 (=0)",
                     "missing_hours": "NaN gaps; windows spanning a gap dropped"},
        "n_clusters": N, "cluster_mean_curtail_mw": dict(zip(names, [float(np.nanmean(curt_g[i])) for i in range(N)])),
        "excluded_clusters": excluded,
        "fleet_mean_mw": {"curtail": float(np.nanmean(fleet_c)), "workload": float(np.nanmean(fleet_w)),
                          "shift_ge1h": float(np.nanmean(fleet_s[1])), "shift_ge4h": float(np.nanmean(fleet_s[4]))},
        "curtail_share_of_workload_pct": float(np.nanmean(fleet_c) / np.nanmean(fleet_w) * 100),
        "headline_fleet_curtail": head,
        "k_surface_fleet_curtail_mw_workload": {f"h{h}": {f"a{a}": float(pick(fl, h, a).k_obs) for a in ALPHAS} for h in HS},
        "k_surface_fleet_curtail_plus_shift_h_mw": {f"h{h}": {f"a{a}": float(pick(fl, h, a).k_obs_curtail_plus_shift_h) for a in ALPHAS} for h in HS},
        "firmness_fleet_curtail": {f"h{h}": {f"a{a}": float(pick(fl, h, a).firm_obs) for a in ALPHAS} for h in HS},
        "effective_counts": {f"h{h}": {f"a{a}": eff_counts(fleet_c, h, a) for a in ALPHAS} for h in HS},
        "portfolio_firmness_h4_a95": port,
        "diversification_gain_h4_a95": port[N]["firm_obs"] / port[1]["firm_obs"],
        "eras_h4_a95": {tag: {"k_obs": float(pick(surf[surf.scope == tag], 4, 0.95).k_obs),
                              "firm_obs": float(pick(surf[surf.scope == tag], 4, 0.95).firm_obs),
                              "mean_mw": float(pick(surf[surf.scope == tag], 4, 0.95)["mean"])} for tag in eras},
        "monthly": monthly, "holdout_coverage": holdout,
        "boot_firmness_nested_h4_a95": {str(n + 1): band(boot_firm_n[:, n]) for n in range(N)},
    }
    surf.to_parquet(os.path.join(OUT, "k_surface.parquet"), index=False)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A6 v2 — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    log("headline: " + json.dumps({k: {kk: v[kk] for kk in ["k_obs_mw", "firm_obs", "const_over_obs", "indep_v2_over_obs"]}
                                   for k, v in head.items()}, indent=1, default=str))

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a, c in zip(ALPHAS, ["#bbb", "tab:blue", "tab:green", "tab:red"]):
        ax.plot(HS, [pick(fl, h, a).k_obs for h in HS], "o-", color=c, label=f"observed, α={a}")
    ax.fill_between(HS, [np.percentile(boot[168][(h, .95)], 5) for h in HS],
                    [np.percentile(boot[168][(h, .95)], 95) for h in HS], color="tab:green", alpha=0.2,
                    label="α=0.95 power-MC × 168 h block bootstrap P5–P95")
    ax.plot(HS, [pick(fl, h, .95).k_const for h in HS], "s--", color="k", label="constant-share (mean-calibrated)")
    ax.plot(HS, [pick(fl, h, .95).k_shuf for h in HS], "^:", color="tab:purple", label="time-shuffled")
    ax.plot(HS, [pick(fl, h, .95).k_indep_v2 for h in HS], "v:", color="tab:orange", label="independent clusters (detrended)")
    ax.axhline(np.nanmean(fleet_c), color="gray", lw=0.8, ls="-.", label=f"mean eligible-workload availability {np.nanmean(fleet_c):.2f} MW")
    ax.set_xscale("log"); ax.set_xticks(HS, [str(h) for h in HS])
    ax.set(xlabel="event duration h (hours)", ylabel="K(α,h), GPU-side MW",
           title=f"eligible-workload availability ({boundary}, {elig}), N={N}")
    ax.legend(fontsize=7)
    savefig(fig, "f1_k_vs_duration.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ns = np.arange(1, N + 1)
    for h, c in zip([1, 4, 24], ["tab:blue", "tab:green", "tab:red"]):
        sub = rnd[(rnd.h == h) & (rnd.alpha == .95)].sort_values("n")
        ax.plot(sub.n, sub.firm_obs, "o-", color=c, label=f"observed, h={h}")
    sub = rnd[(rnd.h == 4) & (rnd.alpha == .95)].sort_values("n")
    ax.plot(sub.n, sub.firm_indep_v2, "v:", color="tab:green", label="independent (detrended) null, h=4")
    ax.plot(sub.n, sub.firm_const, "s--", color="k", label="constant-share, h=4")
    ax.fill_between(ns, np.percentile(boot_firm_n, 5, axis=0), np.percentile(boot_firm_n, 95, axis=0),
                    color="tab:green", alpha=0.15, label="nested top-n boot P5–P95")
    ax.set(xlabel="portfolio size n (logical clusters)", ylabel="K(0.95,h) / mean", ylim=(0, 1.05),
           title="aggregation of eligible-workload availability (all portfolios enumerated)")
    ax.legend(fontsize=7.5)
    savefig(fig, "f2_firmness_vs_portfolio.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    mo = pd.DataFrame(monthly)
    axes[0].plot(mo.start_day, mo.firm_95_4h, "o-")
    axes[0].axvline(REGIME_DAY, color="r", ls="--", lw=1, label=f"day {REGIME_DAY}")
    if standby_day:
        axes[0].axvline(standby_day, color="tab:blue", ls=":", lw=1, label=f"Standby first day {standby_day}")
    axes[0].set(xlabel="30-day window start", ylabel="firmness K(0.95,4h)/mean", ylim=(0, 1.05), title="monthly")
    axes[0].legend(fontsize=8)
    for tag, c in zip(["pre105", "post105", "first_half", "second_half"], ["tab:blue", "tab:red", "gray", "k"]):
        d = surf[surf.scope == tag]
        axes[1].plot(HS, [pick(d, h, .95).firm_obs for h in HS], "o-", color=c, label=tag)
    axes[1].set_xscale("log"); axes[1].set_xticks(HS, [str(h) for h in HS])
    axes[1].set(xlabel="h", ylabel="firmness at α=0.95", ylim=(0, 1.05), title="eras")
    axes[1].legend(fontsize=8)
    savefig(fig, "f3_eras_monthly.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, c in zip(ALPHAS, ["#bbb", "tab:blue", "tab:green", "tab:red"]):
        ax.plot(HS, [holdout[f"h{h}_a{a}"]["coverage_second_half"] for h in HS], "o-", color=c, label=f"target α={a}")
        ax.axhline(a, color=c, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks(HS, [str(h) for h in HS])
    ax.set(xlabel="h", ylabel="Pr[min_h ≥ K_first-half] in second half", ylim=(0, 1.02),
           title="holdout coverage of K estimated on the first half")
    ax.legend(fontsize=8)
    savefig(fig, "f4_holdout_coverage.png")

    # Backlog replay invalidated horizon-specific observed queue-delay shifting
    # as a capacity product, so the old right panel is intentionally removed.
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    M = np.array([[pick(fl, h, a).k_obs / np.nanmean(fleet_c) for a in ALPHAS] for h in HS])
    im = ax.imshow(M, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ALPHAS)), [str(a) for a in ALPHAS]); ax.set_yticks(range(len(HS)), [str(h) for h in HS])
    ax.set(xlabel="reliability α", ylabel="duration h (hours)", title="eligible-workload availability firmness")
    for i in range(len(HS)):
        for j in range(len(ALPHAS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.7 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="K / mean eligible-workload availability")
    savefig(fig, "f5_firmness_heatmap.png")
    log("DONE a6_accreditation v2")


if __name__ == "__main__":
    main()
