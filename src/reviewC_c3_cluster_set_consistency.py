#!/usr/bin/env python
"""C3 (reviewer, combination-basis consistency): confirm which cluster set the
HP-scope sweep (Fig 4c / 02_results HP curve 3.96 / 5.64 / 8.88 MW at 4 h, 95%)
uses, and bound the effect of the clusters the LP threshold (mean eligible >
0.02 MW) drops.

The LP portfolio keeps 13 clusters. Total ASI internal clusters = 16 (E0 lists 3
excluded: 3f39b4b6, 21133261, 65e98ec7). The reviewer asks whether the HP scope
sweep is computed on the same set, and what the LP-excluded clusters contribute
to the HP-expanded envelope.

combined_mw(frac, T, alpha) = P( LP_active + frac*(training_active +
offline_active), T, alpha ), the coincident aggregate over the chosen cluster
set. We reproduce it on (a) the 13 LP-kept clusters and (b) all 16 clusters and
compare against the stored grid value (8.883 MW at frac=1.0, 4 h, 95%).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
import powermodel as pm  # noqa: E402
import flex_decomposition as fd  # noqa: E402

POD = os.path.join(HERE, "..", "2026-09-02_a1-load-reconstruction", "pod_hourly_agg.parquet")
CFG = os.path.join(REPO, "configs", "power_curves.yaml")
OUT = os.path.join(HERE, "c3_cluster_set_consistency")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.95
HORIZONS = (1, 4, 24)
FRACS = (0.25, 0.5, 1.0)


def build():
    cfg = yaml.safe_load(open(CFG))
    pod = pd.read_parquet(POD)
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    cl_codes, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
    ncl = len(cl_uniq)
    layer = pm.assign_layer(pod, cfg.get("eligibility", "central"))
    p0 = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
    parts = pm.row_energy(pod, cfg, p0, spec_codes, spec_uniq)
    act = parts["active"] / 1e6                       # MW-h per row (active only)
    jt = pod["job_type_public"].astype(str).to_numpy()

    lp = layer == "curtail"
    train = jt == "training"
    offl = jt == "offline_inference"
    hp = (pod["priority_class"].astype(str).to_numpy() == "HP")

    def agg(mask):
        M = np.zeros((ncl, nt)); np.add.at(M, (cl_codes[mask], tpod[mask]), act[mask]); return M

    LP = agg(lp)                                      # LP curtailable active
    TR = agg((train) & hp)                            # HP training active
    OF = agg((offl) & hp)                             # HP offline-inference active
    work = np.zeros((ncl, nt)); np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
    good = work.sum(axis=0) > 0
    names = [str(cl_uniq[i])[:8] for i in range(ncl)]
    return LP, TR, OF, good, names, nt


def combined(LP, TR, OF, keep, good, frac, T, alpha):
    A = (LP[keep] + frac * (TR[keep] + OF[keep])).sum(axis=0).astype(float)
    A[~good] = np.nan
    return float(fd.kval(A, T, alpha))


def main():
    LP, TR, OF, good, names, nt = build()
    lp_mean = LP.mean(axis=1)
    kept13 = [i for i in np.argsort(-lp_mean) if lp_mean[i] >= fd.MIN_CLUSTER_MW]
    allcl = list(range(len(names)))
    excluded = [i for i in allcl if i not in kept13]

    res = {"alpha": ALPHA, "min_cluster_mw": fd.MIN_CLUSTER_MW,
           "n_clusters_total": len(names),
           "n_lp_kept": len(kept13),
           "lp_kept_ids": [names[i] for i in kept13],
           "lp_excluded_ids": [names[i] for i in excluded],
           "stored_grid_combined_mw_frac1.0_4h_95": 8.883922163595662,
           "per_excluded_cluster": {}, "combined_mw": {"kept13": {}, "all": {}},
           "excluded_contribution": {}}

    # HP-expanded envelope on both sets
    for T in HORIZONS:
        res["combined_mw"]["kept13"][f"h{T}"] = {
            f"frac{f}": combined(LP, TR, OF, kept13, good, f, T, ALPHA) for f in FRACS}
        res["combined_mw"]["all"][f"h{T}"] = {
            f"frac{f}": combined(LP, TR, OF, allcl, good, f, T, ALPHA) for f in FRACS}

    # per-excluded-cluster diagnostics
    for i in excluded:
        res["per_excluded_cluster"][names[i]] = {
            "lp_active_mean_mw": float(lp_mean[i]),
            "hp_training_active_mean_mw": float(TR[i].mean()),
            "hp_offline_active_mean_mw": float(OF[i].mean()),
            "hp_total_active_mean_mw": float((TR[i] + OF[i]).mean())}

    # contribution of excluded clusters to the 4 h / 95% HP-expanded envelope
    for T in HORIZONS:
        for f in FRACS:
            k13 = res["combined_mw"]["kept13"][f"h{T}"][f"frac{f}"]
            kall = res["combined_mw"]["all"][f"h{T}"][f"frac{f}"]
            res["excluded_contribution"][f"h{T}_frac{f}"] = {
                "kept13_mw": k13, "all_mw": kall,
                "delta_mw": kall - k13,
                "delta_pct": 100 * (kall / k13 - 1) if k13 else None}

    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print(f"total clusters {len(names)}; LP kept {len(kept13)}; excluded {[names[i] for i in excluded]}")
    print("\nExcluded-cluster mean active power (MW):")
    for nm, d in res["per_excluded_cluster"].items():
        print(f"  {nm}: LP {d['lp_active_mean_mw']:.4f}  HP-train {d['hp_training_active_mean_mw']:.4f}  "
              f"HP-offline {d['hp_offline_active_mean_mw']:.4f}  HP-total {d['hp_total_active_mean_mw']:.4f}")
    print("\nHP-expanded envelope combined_mw (kept13 vs all 16):")
    for T in HORIZONS:
        for f in FRACS:
            d = res["excluded_contribution"][f"h{T}_frac{f}"]
            print(f"  T={T:2d}h frac={f}: kept13 {d['kept13_mw']:.3f}  all {d['all_mw']:.3f}  "
                  f"delta {d['delta_mw']:+.4f} MW ({d['delta_pct']:+.2f}%)")
    print("\nStored grid frac=1.0/4h/95% = 8.884; my kept13 4h frac1.0 =",
          round(res["combined_mw"]["kept13"]["h4"]["frac1.0"], 3),
          " all =", round(res["combined_mw"]["all"]["h4"]["frac1.0"], 3))
    print("wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
