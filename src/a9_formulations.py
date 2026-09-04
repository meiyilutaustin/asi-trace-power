#!/usr/bin/env python
"""A9 (2026-09-04, supplementary analysis, 5, 7): published flexibility
formulations vs the measured surface; repeat-event budget; host-side response.

Inputs: a5_v2/envelope_hourly.parquet (+ cluster file), a6_v2/summary.json,
a1_v2/fleet_hourly_power.parquet, agg/pod_hourly_agg.parquet, power_curves.yaml.

(4) Formulations, all evaluated on the SAME hourly series and denominators
    (nameplate = 7-day rolling max facility MW; IT; facility):
    duke_full        100% of load curtailable at any hour (Rethinking Load Growth)
    tier_30_50_20    firm 30% / flexible 50% (shiftable within the day, depth
                     <= 50%, 12 h recovery) / interruptible 20% (<= 5% of annual
                     energy)  (Khanal...Dvorkin 2608.19622, EPRI DCFlex tiers)
    const_mean       constant share = mean eligible share (our mean-calibrated)
    const_K          constant share calibrated to K(0.95,4h)/nameplate (a
                     conservative scalar; the reviewer: need not overstate)
    measured         our hourly eligible curtailment (idle_retained)
    For each: dependable MW at h = 1/4/24 and alpha = 0.95 (running-min
    quantile), on facility-marginal (x1.0) and facility-average (xPUE) bases.
(5) Repeat-event budget: at the level K(0.95,h), the number of non-overlapping
    h-hour windows per week in which the portfolio can deliver K (distribution),
    for h = 1, 4; also for K(0.9,h).
(7) Host-side response of evicted LP pods: measured CPU dynamic part (from the
    aggregate table) plus a bounded allowance for memory/NIC/fans using the
    non-GPU dynamic fraction from published server splits (POLCA: GPUs ~50-60%
    of server power; Acme: 65.7%).  Reported as low/central/high add-ons.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import powermodel as pm  # noqa: E402

AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
A1 = os.environ.get("A1_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a1_v2"))
A5 = os.environ.get("A5_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a5_v2"))
A6 = os.environ.get("A6_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a6_v2"))
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a9_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
HS = [1, 4, 24]
ALPHA = 0.95


def rmin(series, h):
    m = sliding_window_view(series, h).min(axis=1) if h > 1 else series
    return m[~np.isnan(m)]


def kval(series, h, a):
    m = rmin(series, h)
    return float(np.quantile(m, 1 - a)) if len(m) else np.nan


def main():
    cfg = yaml.safe_load(open(CFG))
    pue = cfg["pue"]["base"]
    fl = pd.read_parquet(os.path.join(A1, "fleet_hourly_power.parquet"))
    env = pd.read_parquet(os.path.join(A5, "envelope_hourly.parquet"))
    a6 = json.load(open(os.path.join(A6, "summary.json")))
    good = fl["have"].to_numpy().astype(bool) & env["good"].to_numpy().astype(bool)
    nt = len(fl)
    F = fl["base_mw"].to_numpy().astype(float)                # facility (avg PUE) base params
    IT = F / pue
    C = env["curtail_idle_retained"].to_numpy().astype(float)  # GPU-side eligible curtailment
    W = env["workload"].to_numpy().astype(float)
    cap = pd.Series(F).rolling(168, min_periods=24).max().bfill().to_numpy()
    nan = lambda x: np.where(good, x, np.nan)

    # ---- (4) formulations --------------------------------------------------------
    res = {}
    day = np.arange(nt) // 24
    for basis, factor, denom in [("facility_marginal", 1.0, F), ("facility_avg_pue", pue, F)]:
        c = C * factor
        share_mean = float(np.nanmean(nan(c)) / np.nanmean(nan(F)))
        kshare = kval(nan(c), 4, ALPHA) / float(np.nanmean(nan(cap)))
        forms = {
            "duke_full": nan(F),
            "tier_30_50_20_interruptible_only": nan(0.20 * F),
            "tier_30_50_20_interruptible_plus_flexible": nan(0.70 * F),
            "const_mean": nan(share_mean * F),
            "const_K_nameplate": nan(kshare * cap),
            "measured_idle_retained": nan(c),
            "measured_attributed": nan(env["curtail_attributed"].to_numpy() * factor),
        }
        out = {}
        for name, s in forms.items():
            out[name] = {f"K95_{h}h_mw": kval(s, h, ALPHA) for h in HS}
            out[name]["mean_mw"] = float(np.nanmean(s))
            out[name]["mean_frac_of_facility"] = float(np.nanmean(s) / np.nanmean(nan(F)))
        # tier: interruptible energy cap 5%/year and flexible 12 h recovery -> effective dependable
        # under a 24 h event the flexible tier cannot recover within the event -> only interruptible counts
        out["tier_30_50_20_interruptible_only"]["note"] = "20% at any hour; 5% annual-energy cap not binding for <= 438 h/yr of full use"
        out["tier_30_50_20_interruptible_plus_flexible"]["note"] = "flexible 50% assumes intra-day shift with 12 h recovery: valid for h<=4, NOT for 24 h"
        out["tier_30_50_20_interruptible_plus_flexible"]["K95_24h_mw"] = out["tier_30_50_20_interruptible_only"]["K95_24h_mw"]
        out["ratios_vs_measured"] = {name: {f"{h}h": out[name][f"K95_{h}h_mw"] / out["measured_idle_retained"][f"K95_{h}h_mw"] for h in HS}
                                     for name in forms if name != "measured_idle_retained"}
        out["shares"] = {"mean_eligible_share_of_facility": share_mean, "K95_4h_share_of_nameplate": kshare}
        res[basis] = out

    # ---- (5) repeat-event budget --------------------------------------------------
    rep = {}
    cg = nan(C)
    week = np.arange(nt) // 168
    for h in [1, 4]:
        for a in [0.9, 0.95]:
            K = kval(cg, h, a)
            # greedy non-overlapping windows per week where min over h >= K
            ok = np.zeros(nt, bool)
            m = sliding_window_view(cg, h).min(axis=1) if h > 1 else cg
            i = 0
            counts = {}
            while i < len(m):
                if not np.isnan(m[i]) and m[i] >= K:
                    counts[week[i]] = counts.get(week[i], 0) + 1
                    i += h
                else:
                    i += 1
            per_week = np.array([counts.get(w, 0) for w in range(week.max() + 1) if (week == w).sum() >= 100])
            rep[f"h{h}_a{a}"] = {"K_mw": K, "events_per_week_mean": float(per_week.mean()), "events_per_week_min": int(per_week.min()),
                                 "events_per_week_p10": float(np.percentile(per_week, 10)), "events_per_week_median": float(np.median(per_week)),
                                 "max_possible_per_week": int(168 // h), "weeks": int(len(per_week))}
    # at a contract level of 4-h events, how many per week at 80%/60% of K?
    K4 = kval(cg, 4, 0.95)
    for frac in [0.8, 0.6]:
        m = sliding_window_view(cg, 4).min(axis=1)
        i = 0; counts = {}
        while i < len(m):
            if not np.isnan(m[i]) and m[i] >= frac * K4:
                counts[week[i]] = counts.get(week[i], 0) + 1; i += 4
            else:
                i += 1
        per_week = np.array([counts.get(w, 0) for w in range(week.max() + 1) if (week == w).sum() >= 100])
        rep[f"h4_level{int(frac*100)}pct_of_K95"] = {"level_mw": frac * K4, "events_per_week_mean": float(per_week.mean()),
                                                     "events_per_week_min": int(per_week.min()), "events_per_week_p10": float(np.percentile(per_week, 10))}

    # ---- (7) host-side response --------------------------------------------------
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    layer = pm.assign_layer(pod, cfg.get("eligibility", "central"))
    lp = layer == "curtail"
    h = cfg["host_w_per_core"]
    cpu_dyn = pod.loc[lp, "cpu_used_cores"].to_numpy() * (h["peak"] - h["idle"])
    cpu_dyn_mw = pd.Series(cpu_dyn).groupby(pod.loc[lp, "t"].to_numpy()).sum().reindex(np.arange(nt), fill_value=0).to_numpy() / 1e6
    cpu_dyn_mean = float(np.nanmean(nan(cpu_dyn_mw)))
    gpu_mean = float(np.nanmean(nan(C)))
    # allowance: non-GPU server power is 34-50% of server power (GPU 50-66%); of that, the
    # dynamic (load-following) part for memory/NIC/fans is taken as 10-30%.
    host = {"cpu_dynamic_mw_mean": cpu_dyn_mean, "cpu_dynamic_pct_of_gpu_curtail": cpu_dyn_mean / gpu_mean * 100,
            "allowance_mem_nic_fans_mw": {}}
    for tag, nongpu, dyn in [("low", 0.34, 0.10), ("central", 0.42, 0.20), ("high", 0.50, 0.30)]:
        # non-GPU power attributable to the curtailed GPUs' share of the server, dynamic fraction
        allowance = gpu_mean * (nongpu / (1 - nongpu)) * dyn
        host["allowance_mem_nic_fans_mw"][tag] = {"assumed_nongpu_share": nongpu, "assumed_dynamic_frac": dyn, "mw": allowance,
                                                  "total_host_addon_mw": cpu_dyn_mean + allowance,
                                                  "total_host_addon_pct_of_gpu_curtail": (cpu_dyn_mean + allowance) / gpu_mean * 100}
    host["sources"] = "POLCA (GPUs ~50% provisioned / 60% consumed server power), Acme (GPU 65.7% of server), NLR node idle 420 W"

    summary = {"formulations": res, "repeat_event_budget": rep, "host_side": host,
               "denominators": {"facility_mean_mw": float(np.nanmean(nan(F))), "nameplate_mean_mw": float(np.nanmean(nan(cap))),
                                "it_mean_mw": float(np.nanmean(nan(IT))), "workload_gpu_side_mean_mw": float(np.nanmean(nan(W))),
                                "curtail_idle_retained_gpu_side_mean_mw": gpu_mean}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A9 — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    print(json.dumps({"ratios_marginal": res["facility_marginal"]["ratios_vs_measured"], "repeat": rep, "host": host}, indent=1, default=str))


if __name__ == "__main__":
    main()
