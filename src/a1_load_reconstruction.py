#!/usr/bin/env python
"""A1 (v2, revision 2): reconstruct the six-month hourly power profile.

Input: pod_hourly_agg.parquet + server_hourly_agg.parquet + power_curves.yaml.
Model (src/powermodel.py):
  P_facility = PUE * ( P_gpu_pods + P_gpu_idle_floor + P_host )
GPU pod energy uses the shared row model (online-inference power floor c0,
idle fraction phi, job-type curves from S_b sufficient statistics); Monte
Carlo over {TDP, phi, c0, curve family, XPU TDP, null-util level, host, PUE}.
v2 additions: growth rate from first/last-week means, exact Standby first day,
missing-hour-aware autocorrelation, online-floor share in the summary.
Outputs figures + summary.{json,md} + fleet_hourly_power.parquet into $OUT_DIR.
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
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a1_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(
    os.path.dirname(__file__), "..", "configs", "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def plot_components(base_out, have, hrs):
    """Render the base-parameter IT component panel without the MC sweep."""
    comp = {k: base_out[k] for k in ["gpu_active", "gpu_idle_floor", "host"]}
    dfc = pd.DataFrame({"h": hrs % 24, "have": have, **comp}).query("have")
    gc = dfc.groupby("h").mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.stackplot(gc.index, gc["gpu_active"], gc["gpu_idle_floor"], gc["host"],
                 labels=["Allocated GPU-pod power", "GPU idle floor (unallocated)", "host"], alpha=0.8)
    ax.set(xlabel="hour of day", ylabel="IT power (MW, pre-PUE)",
           title="component decomposition, average day (base params)")
    ax.legend(loc="lower right")
    savefig(fig, "f4_components.png")


def build_inputs():
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    srv = pd.read_parquet(os.path.join(AGG, "server_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    srv["t"] = srv["day"].astype(int) * 24 + srv["hour"].astype(int)
    standby_first_day = None
    sb = pod[(pod["state_public"] == "Standby") & (pod["gpu_hours"] > 0)]
    if len(sb):
        standby_first_day = int(sb["day"].astype(int).min())
    num = ["gpu_hours", "gpu_hours_null_util", "S0", "S25", "S50", "S75",
           "cpu_req_cores", "cpu_used_cores"]
    pod = (pod.groupby(["t", "cluster_id", "gpu_spec_public",
                        "job_type_public", "state_public"],
                       observed=True, dropna=False)[num]
              .sum().reset_index())
    return pod, srv, standby_first_day


def precompute_static(pod, srv):
    st = {}
    occ_df = (pod.groupby(["t", "cluster_id", "gpu_spec_public"],
                          observed=True)["gpu_hours"].sum())
    cap = srv.set_index(["t", "cluster_id", "gpu_spec_public"])["gpu_count"]
    idle_gh = (cap - occ_df.reindex(cap.index, fill_value=0.0)).clip(lower=0)
    st["idle_t"] = idle_gh.index.get_level_values("t").to_numpy()
    st["idle_spec_codes"], st["idle_spec_uniq"] = pd.factorize(
        idle_gh.index.get_level_values("gpu_spec_public"))
    st["idle_gh"] = idle_gh.to_numpy()
    st["pod_spec_codes"], st["pod_spec_uniq"] = pd.factorize(
        pod["gpu_spec_public"])
    st["cap_cores"] = srv.groupby("t")["cpu_capacity_cores"].sum()
    st["used_cores"] = pod[pod["state_public"] != "Standby"].groupby(
        "t")["cpu_used_cores"].sum()
    return st


def compute_power(pod, cfg, p, nt, st):
    parts = pm.row_energy(pod, cfg, p, st["pod_spec_codes"], st["pod_spec_uniq"])
    tpod = pod["t"].to_numpy()
    e_active = np.zeros(nt)
    np.add.at(e_active, tpod, parts["total"])
    # online-inference floor contribution (for the summary): c0 part only
    e_floor_online = np.zeros(nt)
    if p["c0"] > 0:
        on = (pod["job_type_public"].to_numpy() == "online_inference") & \
             (pod["state_public"].to_numpy() != "Standby")
        tdp_row = pm.tdp_of_rows(p, cfg, st["pod_spec_codes"], st["pod_spec_uniq"])
        contrib = tdp_row * (1 - p["idle_frac"]) * p["c0"] * pod["gpu_hours"].to_numpy()
        np.add.at(e_floor_online, tpod[on], contrib[on])
    e_idlefl = np.zeros(nt)
    idle_tdp = pm.tdp_of_rows(p, cfg, st["idle_spec_codes"], st["idle_spec_uniq"])
    np.add.at(e_idlefl, st["idle_t"], st["idle_gh"] * idle_tdp * p["idle_frac"])
    e_host = np.zeros(nt)
    np.add.at(e_host, st["cap_cores"].index.to_numpy(),
              st["cap_cores"].to_numpy() * p["host_idle"])
    np.add.at(e_host, st["used_cores"].index.to_numpy(),
              st["used_cores"].to_numpy() * (p["host_peak"] - p["host_idle"]))
    return {"total": p["pue"] * (e_active + e_idlefl + e_host) / 1e6,
            "gpu_active": e_active / 1e6, "gpu_idle_floor": e_idlefl / 1e6,
            "host": e_host / 1e6, "online_floor_part": e_floor_online / 1e6,
            "it": (e_active + e_idlefl + e_host) / 1e6}


def nan_autocorr(x, maxlag):
    """Autocorrelation ignoring NaN pairs (missing hour kept as a gap)."""
    x = x - np.nanmean(x)
    out = np.zeros(maxlag)
    for k in range(maxlag):
        a, b = x[:len(x) - k], x[k:]
        m = ~np.isnan(a) & ~np.isnan(b)
        out[k] = np.sum(a[m] * b[m]) / np.sum(x[~np.isnan(x)] ** 2) * (len(x) / m.sum())
    return out / out[0]


def main():
    cfg = yaml.safe_load(open(CFG))
    pod, srv, standby_first_day = build_inputs()
    specs = sorted(set(pod["gpu_spec_public"].dropna()) |
                   set(srv["gpu_spec_public"].dropna()))
    nt = int(max(pod["t"].max(), srv["t"].max())) + 1
    log(f"agg rows: pod={len(pod)}, srv={len(srv)}, hours={nt}, specs={specs}")
    rng = np.random.default_rng(cfg["mc_seed"])
    st = precompute_static(pod, srv)
    base = pm.draw_params(cfg, rng, specs, base=True)
    base_out = compute_power(pod, cfg, base, nt, st)
    if os.environ.get("A1_COMPONENTS_ONLY") == "1":
        have = np.zeros(nt, bool)
        have[srv["t"].unique()] = True
        pod_gh = pod.groupby("t")["gpu_hours"].sum()
        have[pod_gh.index[pod_gh <= 0].to_numpy()] = False
        have[np.setdiff1d(np.arange(nt), pod["t"].unique())] = False
        plot_components(base_out, have, np.arange(nt))
        log("DONE A1 components-only refresh")
        return
    ns = int(cfg["mc_samples"])
    mc = np.zeros((ns, nt))
    mc_it = np.zeros((ns, nt))
    for i in range(ns):
        o = compute_power(pod, cfg, pm.draw_params(cfg, rng, specs), nt, st)
        mc[i], mc_it[i] = o["total"], o["it"]
        if (i + 1) % 50 == 0:
            log(f"MC {i+1}/{ns}")
    p5, p50, p95 = np.percentile(mc, [5, 50, 95], axis=0)
    it50 = np.percentile(mc_it, 50, axis=0)

    have = np.zeros(nt, bool)
    have[srv["t"].unique()] = True
    pod_gh = pod.groupby("t")["gpu_hours"].sum()
    have[pod_gh.index[pod_gh <= 0].to_numpy()] = False
    have[np.setdiff1d(np.arange(nt), pod["t"].unique())] = False
    hrs = np.arange(nt)
    tot = base_out["total"]

    p5m, p50m, p95m = (np.where(have, a, np.nan) for a in (p5, p50, p95))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.fill_between(hrs / 24, p5m, p95m, alpha=0.3, lw=0, label="MC P5–P95")
    ax.plot(hrs / 24, p50m, lw=0.4, label="median")
    ax.set(xlabel="day", ylabel="facility power (MW)",
           title="ASI fleet reconstructed hourly power, 6 months (v2: online floor)")
    ax.legend()
    savefig(fig, "f1_fleet_load_band.png")

    prof = pd.DataFrame({"h": hrs % 24, "p50": p50, "p5": p5, "p95": p95,
                         "have": have}).query("have")
    g = prof.groupby("h").mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(g.index, g["p5"], g["p95"], alpha=0.3, lw=0)
    ax.plot(g.index, g["p50"], marker="o")
    ax.set(xlabel="hour of day", ylabel="facility power (MW)", title="average daily profile")
    savefig(fig, "f2_daily_profile.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, lab in [(p50, "median"), (p5, "P5"), (p95, "P95")]:
        v = np.sort(arr[have])[::-1]
        ax.plot(np.linspace(0, 100, len(v)), v, label=lab)
    ax.set(xlabel="% of hours", ylabel="facility power (MW)", title="load duration curve")
    ax.legend()
    savefig(fig, "f3_load_duration.png")

    plot_components(base_out, have, hrs)

    xnan = np.where(have, p50, np.nan)
    ac = nan_autocorr(xnan, 24 * 15)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(ac)) / 24, ac)
    for d in range(1, 15):
        ax.axvline(d, color="gray", lw=0.4, alpha=0.5)
    ax.set(xlabel="lag (days)", ylabel="autocorrelation (missing hour excluded)",
           title="periodicity of the reconstructed load")
    savefig(fig, "f5_autocorr.png")

    v = p50[have]
    daily = pd.DataFrame({"d": hrs[have] // 24, "v": v}).groupby("d")["v"]
    amp = (daily.max() - daily.min())
    dmean = daily.mean()
    first_week = float(dmean.loc[0:6].mean())
    last_week = float(dmean.loc[dmean.index.max() - 6:].mean())
    summary = {
        "hours_covered": int(have.sum()),
        "mean_mw": {"p50": float(v.mean()), "p5": float(p5[have].mean()),
                    "p95": float(p95[have].mean()), "base": float(tot[have].mean())},
        "mean_it_mw_p50": float(it50[have].mean()),
        "peak_mw_p50": float(v.max()), "min_mw_p50": float(v.min()),
        "load_factor_p50": float(v.mean() / v.max()),
        "peak_to_trough_p50": float(v.max() / v.min()),
        "growth": {"first_week_mean_mw": first_week, "last_week_mean_mw": last_week,
                   "growth_pct": float((last_week / first_week - 1) * 100)},
        "avg_daily_amplitude_mw": float(amp.mean()),
        "avg_daily_amplitude_pct_of_mean": float(amp.mean() / v.mean() * 100),
        "daily_profile_peak_hour": int(g["p50"].idxmax()),
        "daily_profile_trough_hour": int(g["p50"].idxmin()),
        "daily_profile_swing_pct": float((g["p50"].max() - g["p50"].min()) / g["p50"].mean() * 100),
        "autocorr_24h": float(ac[24]), "autocorr_7d": float(ac[24 * 7]),
        "component_share_pct_base": {
            k: float(np.nansum(vv[have]) / np.nansum(sum(comp.values())[have]) * 100)
            for k, vv in comp.items()},
        "online_floor_share_of_it_pct_base": float(
            np.nansum(base_out["online_floor_part"][have]) / np.nansum(base_out["it"][have]) * 100),
        "standby_first_day": standby_first_day,
        "base_params": {k: v_ for k, v_ in base.items() if k != "tdp"},
        "config": {"online_floor_frac": cfg.get("online_floor_frac"),
                   "curve_families": cfg["curve_families"]},
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A1 load reconstruction (v2) — summary\n\n")
        for k, v_ in summary.items():
            f.write(f"- **{k}**: {v_}\n")
    pd.DataFrame({"t": hrs, "have": have, "p5_mw": p5, "p50_mw": p50,
                  "p95_mw": p95, "base_mw": tot, "it_p50_mw": it50}).to_parquet(
        os.path.join(OUT, "fleet_hourly_power.parquet"), index=False)
    log("DONE a1_load_reconstruction v2")


if __name__ == "__main__":
    main()
