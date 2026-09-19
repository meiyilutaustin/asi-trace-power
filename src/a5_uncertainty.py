#!/usr/bin/env python
"""A5b: Monte Carlo uncertainty bands on the flexibility envelope.

Reuses the A1 power-model parameter sampling (TDP, idle_frac, curve family,
XPU TDP, null-util level) and, for each draw, recomputes the four-layer
envelope's key capacity metrics. Reports P5/P50/P95 bands so paper D can show
that the DR headline is robust to the power model, not a single-point claim.

Cheap: operates on the A1 aggregate table only. Outputs bands.json + a banded
average-day figure into $OUT_DIR.
"""
import json
import os

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGG = os.environ.get("AGG_DIR", "/project/mli30/mli30/asi-trace/agg")
DATA = os.environ.get("DATA_DIR", "/project/mli30/mli30/asi-trace/data")
OUT = os.environ.get("OUT_DIR", "/project/mli30/mli30/asi-trace/a5_out")
CFG = os.environ.get("POWER_CFG", os.path.join(
    os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
TAUS = [1, 4]
SHIFTABLE = ["offline_inference", "training"]


def log(m):
    print(m, flush=True)


def draw_params(cfg, rng, specs, base=False):
    p = {}
    ir = cfg["idle_frac"]
    p["idle_frac"] = ir["base"] if base else rng.uniform(*ir["range"])
    nr = cfg["null_util_g"]
    p["null_g"] = nr["base"] if base else rng.uniform(*nr["range"])
    p["family"] = "anchored" if (base or rng.random() < 0.5) else "linear"
    tdp = dict(cfg["gpu_tdp_w"])
    ru = cfg["tdp_rel_uncertainty"]
    lo, hi = cfg["xpu_tdp_range_w"]
    p["tdp"] = {}
    for s in specs:
        if s in tdp:
            p["tdp"][s] = tdp[s] if base else tdp[s] * rng.uniform(1 - ru, 1 + ru)
        elif str(s).startswith("XPU"):
            p["tdp"][s] = 0.5 * (lo + hi) if base else rng.uniform(lo, hi)
        else:
            p["tdp"][s] = cfg["default_tdp_w"]
    return p


def row_power(pod, cfg, p, spec_codes, spec_uniq):
    fam = cfg["curve_families"][p["family"]]
    tdp_row = np.array([p["tdp"].get(s, cfg["default_tdp_w"])
                        for s in spec_uniq])[spec_codes]
    S = pod[["S0", "S25", "S50", "S75"]].to_numpy()
    jt = pod["job_type_public"].to_numpy()
    cw = np.zeros((len(pod), 4))
    for t_, kv in fam.items():
        mask = jt == t_ if t_ != "default" else ~np.isin(
            jt, [k for k in fam if k != "default"])
        slopes = np.diff(np.asarray(kv, float)) / np.diff(KNOTS)
        cw[mask] = np.concatenate([[slopes[0]], np.diff(slopes)])
    w = pod["gpu_hours"].to_numpy()
    wn = pod["gpu_hours_null_util"].to_numpy()
    pw = tdp_row * (p["idle_frac"] * w
                    + (1 - p["idle_frac"]) * ((S * cw).sum(axis=1) + wn * p["null_g"]))
    standby = (pod["state_public"] == "Standby").to_numpy()
    pw[standby] = (tdp_row * p["idle_frac"] * w)[standby]
    return pw / 1e6


def tolerance():
    js = pd.read_parquet(
        os.path.join(DATA, "asi_opensource_job_execution_summary"),
        columns=["gpu_request", "duration_hours", "schedule_delay_sec",
                 "job_type_public", "priority_class"])
    js["w"] = (js["gpu_request"].fillna(0).clip(lower=0)
               * js["duration_hours"].fillna(0).clip(lower=0))
    js["dh"] = js["schedule_delay_sec"].fillna(0).clip(lower=0) / 3600
    out = {}
    for (jt, pr), g in js.groupby(["job_type_public", "priority_class"],
                                  observed=True):
        tot = g["w"].sum()
        if tot > 0:
            out[(jt, pr)] = {t: float(g.loc[g["dh"] >= t, "w"].sum() / tot)
                             for t in TAUS}
    return out


def main():
    cfg = yaml.safe_load(open(CFG))
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])

    jt = pod["job_type_public"].astype(str)
    pr = pod["priority_class"].astype(str)
    standby = pod["state_public"].astype(str) == "Standby"
    layer = np.where(standby, "standby",
             np.where(pr == "LP", "curtail",
             np.where((pr == "HP") & jt.isin(SHIFTABLE), "shift", "floor")))
    pod["layer"] = layer
    tpod = pod["t"].to_numpy()
    tol = tolerance()
    s1 = pod.apply(lambda r: tol.get((r["job_type_public"],
                   r["priority_class"]), {t: 0 for t in TAUS})[1]
                   if r["layer"] == "shift" else 0.0, axis=1).to_numpy()

    rng = np.random.default_rng(cfg["mc_seed"])
    ns = int(cfg.get("mc_samples", 300))
    hour_of_day = (np.arange(nt) % 24)
    metrics = {k: np.zeros(ns) for k in
               ["curtail_mean", "curtail_p10", "shift1_mean",
                "dr_mean", "dr_pct", "floor_mean"]}
    avgday = {L: np.zeros((ns, 24)) for L in ["floor", "shift", "curtail"]}

    for i in range(ns):
        p = draw_params(cfg, rng, specs, base=(i == 0))
        pw = row_power(pod, cfg, p, spec_codes, spec_uniq)
        tot_by_layer = {}
        for L in ["floor", "shift", "curtail", "standby"]:
            arr = np.zeros(nt)
            m = pod["layer"].to_numpy() == L
            np.add.at(arr, tpod[m], pw[m])
            tot_by_layer[L] = arr
        shift1 = np.zeros(nt)
        m = pod["layer"].to_numpy() == "shift"
        np.add.at(shift1, tpod[m], (pw * s1)[m])
        good = sum(tot_by_layer.values()) > 0
        workload = sum(tot_by_layer.values())
        dr = tot_by_layer["curtail"] + shift1
        metrics["curtail_mean"][i] = tot_by_layer["curtail"][good].mean()
        metrics["curtail_p10"][i] = np.percentile(tot_by_layer["curtail"][good], 10)
        metrics["shift1_mean"][i] = shift1[good].mean()
        metrics["dr_mean"][i] = dr[good].mean()
        metrics["dr_pct"][i] = dr[good].mean() / workload[good].mean() * 100
        metrics["floor_mean"][i] = tot_by_layer["floor"][good].mean()
        for L in ["floor", "shift", "curtail"]:
            for h in range(24):
                sel = good & (hour_of_day == h)
                avgday[L][i, h] = tot_by_layer[L][sel].mean()
        if (i + 1) % 50 == 0:
            log(f"MC {i+1}/{ns}")

    def band(a):
        return {"p5": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "base": float(a[0])}
    bands = {k: band(v) for k, v in metrics.items()}
    json.dump(bands, open(os.path.join(OUT, "uncertainty_bands.json"), "w"),
              indent=2)
    log("bands: " + json.dumps(bands, indent=1))

    # banded average-day stack
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(24)
    base_stack = np.zeros(24)
    cols = {"floor": "#666", "shift": "tab:orange", "curtail": "tab:green"}
    for L in ["floor", "shift", "curtail"]:
        p50 = np.percentile(avgday[L], 50, axis=0)
        ax.fill_between(xs, base_stack, base_stack + p50, color=cols[L],
                        alpha=0.85, label=L)
        base_stack = base_stack + p50
    dr_lo = np.percentile(metrics["dr_pct"], 5)
    dr_hi = np.percentile(metrics["dr_pct"], 95)
    ax.set(xlabel="hour of day", ylabel="MW (P50 layers)",
           title=f"envelope with MC power model\nDR capacity = "
                 f"{bands['dr_pct']['p50']:.1f}% "
                 f"(P5–P95 {dr_lo:.1f}–{dr_hi:.1f}%)")
    ax.legend()
    fig.savefig(os.path.join(OUT, "f6_envelope_band.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log("wrote f6_envelope_band.png")
    log("DONE a5_uncertainty")


if __name__ == "__main__":
    main()
