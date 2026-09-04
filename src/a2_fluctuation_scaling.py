#!/usr/bin/env python
"""A2 (v2, revision 2 / /): fluctuation scaling and
cross-cluster synchrony of reconstructed GPU power.

Levels: (cluster, job_type, priority) -> (cluster, job_type) -> cluster -> fleet.
Per unit: mu = mean, sigma = std of the detrended residual (169 h centred
rolling mean), CV = sigma/mu; Taylor fit log sigma = beta log mu + b.
Synchrony ratio child->parent = std(sum of child residuals) / sqrt(sum var).

v2 additions:
  * three residual definitions, all reported:
      raw          detrended residual (v1)
      deseason     residual minus its hour-of-day x day-of-week mean profile
                   (removes the shared human calendar)
      capnorm      series divided by its own 168 h rolling mean before
                   detrending (removes shared growth), then deseasonalised
  * bootstrap CI for beta (unit resampling), jackknife / leave-one-cluster-out
    intervals for the cluster->fleet synchrony ratio
  * the equal-weight identity sqrt(1 + (n-1) rho_bar) reported next to the
    measured cluster->fleet ratio (the reviewer #23)
  * excluded clusters listed with their power share 
  * U-shape: bootstrap CI for c2 
  * missing hours are NaN gaps ; power from src/powermodel.py (online floor)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import powermodel as pm  # noqa: E402

AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a2_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
MIN_HOURS = 1500
MIN_MEAN_MW = 0.01
ROLL = 169
NBOOT = 500
LEVELS = [("cluster_type_prio", ["cluster_id", "job_type_public", "priority_class"]),
          ("cluster_type", ["cluster_id", "job_type_public"]),
          ("cluster", ["cluster_id"]), ("fleet", [])]
RESIDS = ["raw", "deseason", "capnorm"]
CAP_FLOOR_FRAC = 0.2   # guard for the capacity-normalised residual (see r4_covariance_decomposition.py)


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def unit_series(pod, keys, nt, good_t):
    if keys:
        g = pod.groupby(keys + ["t"], observed=True)["p_mw"].sum().unstack(
            keys if len(keys) > 1 else keys[0]).reindex(np.arange(nt))
    else:
        g = pod.groupby("t")["p_mw"].sum().reindex(np.arange(nt)).to_frame("fleet")
    g[~good_t] = np.nan
    return g


def residuals(wide, kind):
    """Residual matrix (t x units) under one definition."""
    trend = wide.rolling(ROLL, center=True, min_periods=24).mean()
    r = wide - trend
    if kind == "capnorm":
        # express each unit's residual at a COMMON capacity level:
        #   r_i(t) = (w_i - trend_i) * mean_i / cap_i(t)
        # Guarded: hours where the unit's 168 h mean falls below 20% of its own
        # median (or 0.05 MW) are masked. The earlier unguarded form divided the
        # LEVEL by cap, and units that idle to ~0 MW then produced ratios up to
        # 168x and carried 97% of the normalised variance -> an earlier unguarded version produced that artefact (see r4_covariance_decomposition.py).
        cap = wide.rolling(168, min_periods=24).mean().bfill()
        floor = np.maximum(cap.median() * CAP_FLOOR_FRAC, 0.05)
        valid = cap.ge(floor, axis=1)
        r = (r * (wide.mean() / cap.where(valid))).where(valid)
    if kind in ("deseason", "capnorm"):
        t = np.arange(len(r))
        key = pd.Series((t % 24) + 24 * ((t // 24) % 7), index=r.index)
        prof = r.groupby(key).transform("mean")
        r = r - prof
    return r


def unit_stats(wide, kind):
    resid = residuals(wide, kind)     # MW for every kind
    mu = wide.mean()
    sigma = resid.std()
    return pd.DataFrame({"mu": mu, "sigma": sigma, "cv": sigma / mu,
                         "hours": wide.notna().sum(), "peak": wide.max()}), resid


def taylor_fit(stats, rng=None, nboot=0):
    s = stats.dropna()
    s = s[(s["mu"] > 0) & (s["sigma"] > 0)]
    if len(s) < 3:
        return None
    x, y = np.log10(s["mu"].to_numpy()), np.log10(s["sigma"].to_numpy())
    beta, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    out = {"beta": float(beta), "intercept": float(b), "r2": float(r * r), "n_units": int(len(s))}
    if rng is not None and nboot and len(s) >= 4:
        bs = []
        for _ in range(nboot):
            i = rng.integers(0, len(s), len(s))
            if len(np.unique(x[i])) < 2:
                continue
            bs.append(np.polyfit(x[i], y[i], 1)[0])
        out["beta_ci90"] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    return out


def sync_rows(child_stats, child_resid, pkeys):
    rows = []
    groups = (child_stats.groupby([child_stats.index.get_level_values(i) for i in range(len(pkeys))])
              if pkeys else [("fleet", child_stats)])
    for pk, sub in groups:
        if isinstance(pk, tuple) and len(pk) == 1:
            pk = pk[0]
        if len(sub) < 2:
            continue
        agg = child_resid[sub.index].sum(axis=1)
        actual = float(agg.std())
        indep = float(np.sqrt((sub["sigma"] ** 2).sum()))
        rows.append({"parent": str(pk), "n_children": int(len(sub)), "sync_ratio": actual / indep})
    return rows


def main():
    cfg = yaml.safe_load(open(CFG))
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tot_gh = pod.groupby("t")["gpu_hours"].sum().reindex(np.arange(nt), fill_value=0)
    good_t = (tot_gh > 0).to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    rng = np.random.default_rng(2)
    log(f"rows {len(pod)}, hours {int(good_t.sum())}/{nt}")
    summary = {"config": {"roll_h": ROLL, "min_hours": MIN_HOURS, "min_mean_mw": MIN_MEAN_MW,
                          "nboot": NBOOT, "residuals": RESIDS}}
    results = {}
    for family in ["anchored", "linear"]:
        p = pm.draw_params(cfg, rng, specs, base=True)
        p["family"] = family
        pod["p_mw"] = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)["total"] / 1e6
        results[family] = {}
        wides = {name: unit_series(pod, keys, nt, good_t) for name, keys in LEVELS}
        for kind in RESIDS:
            lv = {}
            for name, keys in LEVELS:
                stats, resid = unit_stats(wides[name], kind)
                resid = resid.dropna(axis=1, how="all")
                stats = stats.loc[[i for i in stats.index if i in resid.columns]] if len(resid.columns) else stats
                keep = stats[(stats["hours"] >= MIN_HOURS) & (stats["mu"] >= MIN_MEAN_MW)]
                excl = stats.drop(keep.index)
                fit = taylor_fit(keep, rng, NBOOT if kind == "raw" else 0)
                lv[name] = {"fit": fit, "stats": keep, "resid": resid[keep.index], "wide": wides[name][keep.index],
                            "excluded": [{"unit": str(i), "mu_mw": float(r.mu), "hours": int(r.hours)}
                                         for i, r in excl.iterrows()],
                            "excluded_power_share_pct": float(excl["mu"].sum() / stats["mu"].sum() * 100) if stats["mu"].sum() > 0 else 0.0}
            sync = {}
            for (child, ckeys), (parent, pkeys) in zip(LEVELS[:-1], LEVELS[1:]):
                sync[f"{child}->{parent}"] = sync_rows(lv[child]["stats"], lv[child]["resid"], pkeys)
            # cluster->fleet: jackknife / leave-one-cluster-out and rho identity
            cs, cr = lv["cluster"]["stats"], lv["cluster"]["resid"]
            n = len(cs)
            full = sync["cluster->fleet"][0]["sync_ratio"] if sync["cluster->fleet"] else None
            loo = []
            for c in cs.index:
                sub = cs.drop(c)
                loo.append(float(cr[sub.index].sum(axis=1).std() / np.sqrt((sub["sigma"] ** 2).sum())))
            corr = cr.corr()
            iu = np.triu_indices(n, 1)
            rho_bar = float(np.nanmean(corr.values[iu]))
            jk = np.array(loo)
            jk_se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())) if n > 1 else None
            lv["cluster_fleet_sync"] = {
                "ratio": full, "n_clusters": int(n), "loo_min": float(jk.min()), "loo_max": float(jk.max()),
                "jackknife_se": jk_se, "ci90_jackknife": [full - 1.645 * jk_se, full + 1.645 * jk_se] if jk_se else None,
                "rho_bar_pairwise": rho_bar, "rho_median_pairwise": float(np.nanmedian(corr.values[iu])),
                "equal_weight_identity_sqrt_1_plus_n1_rho": float(np.sqrt(max(0.0, 1 + (n - 1) * rho_bar)))}
            lv["sync"] = sync
            results[family][kind] = lv
            log(f"[{family}/{kind}] beta: " + ", ".join(f"{nm}={lv[nm]['fit']['beta']:.2f}" for nm, _ in LEVELS if lv[nm]["fit"])
                + f"; cluster->fleet sync {full:.2f}, rho_bar {rho_bar:.2f}")

    gh = pod.groupby(["cluster_id", "job_type_public"], observed=True)["gpu_hours"].sum().unstack(fill_value=0)
    tshare = (gh.get("training", 0) / gh.sum(axis=1)).rename("train_share")
    an = results["anchored"]["raw"]
    cstats = an["cluster"]["stats"].join(tshare).dropna(subset=["cv", "train_share"])
    qfit = np.polyfit(cstats["train_share"], cstats["cv"], 2)
    c2b = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(cstats), len(cstats))
        try:
            c2b.append(np.polyfit(cstats["train_share"].to_numpy()[i], cstats["cv"].to_numpy()[i], 2)[0])
        except np.linalg.LinAlgError:
            pass
    summary["u_shape_quadfit"] = {"c2": float(qfit[0]), "c1": float(qfit[1]), "c0": float(qfit[2]),
                                  "n_clusters": int(len(cstats)),
                                  "c2_ci90_bootstrap": [float(np.percentile(c2b, 5)), float(np.percentile(c2b, 95))],
                                  "c2_ci_excludes_zero": bool(np.percentile(c2b, 5) > 0 or np.percentile(c2b, 95) < 0)}

    for family in ["anchored", "linear"]:
        summary[family] = {}
        for kind in RESIDS:
            lv = results[family][kind]
            summary[family][kind] = {
                "taylor": {name: lv[name]["fit"] for name, _ in LEVELS},
                "n_units": {name: int(len(lv[name]["stats"])) for name, _ in LEVELS},
                "fleet_cv": float(lv["fleet"]["stats"]["cv"].iloc[0]),
                "sync_median": {k: float(np.median([r["sync_ratio"] for r in v])) if v else None for k, v in lv["sync"].items()},
                "cluster_fleet_sync": lv["cluster_fleet_sync"],
                "excluded_clusters": lv["cluster"]["excluded"],
                "excluded_cluster_power_share_pct": lv["cluster"]["excluded_power_share_pct"]}
    sy = pd.DataFrame(an["sync"]["cluster_type->cluster"]).set_index("parent").join(tshare)
    try:
        sy["tercile"] = pd.qcut(sy["train_share"], 3, labels=["low", "mid", "high"])
        summary["sync_by_train_share"] = {str(k): {"median_sync_ratio": float(v["sync_ratio"].median()), "n": int(len(v))}
                                          for k, v in sy.groupby("tercile", observed=True)}
    except ValueError as e:
        summary["sync_by_train_share"] = f"qcut failed: {e}"

    # ---- figures ---------------------------------------------------------------
    colors = dict(zip([n for n, _ in LEVELS], plt.cm.viridis(np.linspace(0, 0.9, len(LEVELS)))))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, _ in LEVELS:
        s = an[name]["stats"]
        lab = f'{name} (β={an[name]["fit"]["beta"]:.2f}' + (f', CI {an[name]["fit"]["beta_ci90"][0]:.2f}–{an[name]["fit"]["beta_ci90"][1]:.2f})' if an[name]["fit"] and "beta_ci90" in an[name]["fit"] else ")") if an[name]["fit"] else name
        ax.scatter(s["mu"], s["sigma"], s=14, alpha=0.6, color=colors[name], label=lab)
    lo = min(an[n]["stats"]["mu"].min() for n, _ in LEVELS); hi = max(an[n]["stats"]["mu"].max() for n, _ in LEVELS)
    xs = np.array([lo, hi])
    ax.plot(xs, 0.05 * xs, "k--", lw=0.8, label="slope 1 (full sync)")
    ax.plot(xs, 0.05 * lo ** 0.5 * (xs / lo) ** 0.5, "k:", lw=0.8, label="slope 1/2 (independent benchmark)")
    ax.set(xscale="log", yscale="log", xlabel="unit mean power μ (MW)", ylabel="detrended σ (MW)",
           title="Taylor fluctuation scaling across aggregation levels (raw residual)")
    ax.legend(fontsize=7.5)
    savefig(fig, "f1_taylor_levels.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, _ in LEVELS:
        s = an[name]["stats"]
        ax.scatter(s["mu"], s["cv"] * 100, s=14, alpha=0.6, color=colors[name], label=name)
    ax.set(xscale="log", yscale="log", xlabel="unit mean power μ (MW)", ylabel="CV of detrended power (%)",
           title="volatility vs size")
    ax.legend(fontsize=8)
    savefig(fig, "f2_cv_vs_size.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cstats["train_share"] * 100, cstats["cv"] * 100, s=25)
    xs = np.linspace(0, cstats["train_share"].max(), 100)
    ci = summary["u_shape_quadfit"]["c2_ci90_bootstrap"]
    ax.plot(xs * 100, np.polyval(qfit, xs) * 100, "r-", lw=1, label=f"quadratic fit c2={qfit[0]:.2f} (90% CI {ci[0]:.2f}–{ci[1]:.2f})")
    ax.set(xlabel="training share of GPU-hours (%)", ylabel="cluster CV (%)", title="composition vs volatility (exploratory, n=%d)" % len(cstats))
    ax.legend(fontsize=8)
    savefig(fig, "f3_ushape.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    pairs = [k for k, _ in zip(an["sync"].keys(), range(3))]
    for j, kind in enumerate(RESIDS):
        vals = [summary["anchored"][kind]["sync_median"][k] for k in pairs]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, label=f"residual: {kind}")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xticks(x, [k.replace("->", "→") for k in pairs], fontsize=8)
    ax.set(ylabel="median synchrony ratio (1 = independent)", title="synchrony by level: raw vs deseasonalised vs capacity-normalised")
    ax.legend(fontsize=8)
    savefig(fig, "f4_sync_by_residual.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    top = an["cluster"]["stats"].nlargest(4, "mu").index
    for c in top:
        axes[0].plot(np.arange(nt) / 24, an["cluster"]["wide"][c], lw=0.4, label=str(c)[:10])
    axes[0].set(ylabel="cluster MW"); axes[0].legend(fontsize=7, ncol=4)
    axes[1].plot(np.arange(nt) / 24, an["fleet"]["wide"]["fleet"], lw=0.4, color="k")
    axes[1].set(xlabel="day", ylabel="fleet MW (GPU-attributed)")
    savefig(fig, "f5_series_examples.png")

    an["cluster"]["stats"].join(tshare).to_parquet(os.path.join(OUT, "cluster_stats.parquet"))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A2 v2 — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    log("DONE a2_fluctuation_scaling v2")


if __name__ == "__main__":
    main()
