#!/usr/bin/env python
"""A3 (v2, 2026-09-04): scheduler shifting from the run-on-arrival counterfactual,
with the v2 power model, the post-coverage-jump window, and an
energy-constrained backlog replay of delay-based deferral (the scheduler supplementary analysis).

Counterfactual: every matched pod starts schedule_delay earlier ("run on
arrival"), same flat power footprint. D(t) = actual - counterfactual.
v2 changes:
  * W per busy GPU by (job_type, accelerator family) from src/powermodel.py
    (online-inference floor included);
  * all battery statistics reported for the FULL window and for the window
    after the job-summary coverage jump (day >= COVER_DAY, where coverage of
    spans is ~100%); the pre-jump window is unreliable (A3 join audit);
  * backlog replay: a product of horizon h invoked at hour t may defer the
    start of shiftable pods (HP training / offline_inference) that ARRIVE in
    [t, t+1) and that in reality waited >= h (observed delay); the deferred
    power returns h hours later (energy neutral).  Reports the hourly
    deferrable arrival power S_h(t), its mean, K(0.95, h) and the shifted
    energy per invocation, for h = 1, 4.  This is the observed-delay
    counterpart of the class-average s_tau scaling used in A5.
Inputs: agg/pod_spans_b*.parquet, job_execution_summary, pod_hourly_agg.
"""
import json
import os
import sys
import zlib

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
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a3_v2"))
CFG = os.environ.get("POWER_CFG", os.path.join(os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
EDGE_DAYS = 14
COVER_DAY = int(os.environ.get("A3_COVER_DAY", 109))   # first day with ~100% span coverage
HS = [1, 4]


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def watts_per_gpu_hour(cfg):
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    p = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
    e = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)["total"]
    d = pd.DataFrame({"jt": pod["job_type_public"].astype(str),
                      "xpu": pod["gpu_spec_public"].astype(str).str.startswith("XPU"),
                      "p": e, "gh": pod["gpu_hours"]})
    g = d.groupby(["jt", "xpu"]).sum()
    return (g["p"] / g["gh"]).to_dict()


def add_steps(arr, pos, height):
    i = np.floor(pos).astype(int)
    frac = pos - i
    np.add.at(arr, i, height * (1 - frac))
    np.add.at(arr, i + 1, height * frac)


def build_series(df, nt):
    a = np.zeros(nt + 2); c = np.zeros(nt + 2)
    h = df["h"].to_numpy()
    add_steps(a, df["t_min"].to_numpy().astype(float), h)
    add_steps(a, (df["t_max"].to_numpy() + 1).astype(float), -h)
    add_steps(c, df["cf_start"].to_numpy(), h)
    add_steps(c, df["cf_end"].to_numpy(), -h)
    return np.cumsum(a)[:nt], np.cumsum(c)[:nt]


def kval(series, h, alpha):
    m = sliding_window_view(series, h).min(axis=1) if h > 1 else series
    m = m[~np.isnan(m)]
    return float(np.quantile(m, 1 - alpha)) if len(m) else np.nan


def battery_stats(D, A, good, days, tag):
    sel = good
    E = np.cumsum(np.where(sel, D, 0.0))
    dfE = pd.DataFrame({"d": days[sel], "E": E[sel]})
    swing = dfE.groupby("d")["E"].agg(lambda x: x.max() - x.min())
    mean_load = float(A[sel].mean())
    return {"window": tag, "hours": int(sel.sum()), "mean_actual_mw": mean_load,
            "p99_discharge_mw": float(np.percentile(D[sel], 99)), "p99_charge_mw": float(-np.percentile(D[sel], 1)),
            "p99_discharge_pct_of_load": float(np.percentile(D[sel], 99) / mean_load * 100),
            "p99_charge_pct_of_load": float(-np.percentile(D[sel], 1) / mean_load * 100),
            "max_abs_mw": float(np.abs(D[sel]).max()), "E_range_mwh": float(E[sel].max() - E[sel].min()),
            "daily_cycle_median_mwh": float(swing.median()), "daily_cycle_p90_mwh": float(swing.quantile(0.9)),
            "daily_cycle_med_pct_of_daily_energy": float(swing.median() / (mean_load * 24) * 100)}


def main():
    cfg = yaml.safe_load(open(CFG))
    wpg = watts_per_gpu_hour(cfg)
    log(f"W per busy GPU (type, xpu): { {f'{k[0]}|{int(k[1])}': round(v) for k, v in wpg.items()} }")
    js = pd.read_parquet(os.path.join(DATA, "asi_opensource_job_execution_summary"),
                         columns=["pod_id", "gpu_request", "duration_hours", "schedule_delay_sec",
                                  "job_type_public", "priority_class", "gpu_spec_public"])
    js = js.drop_duplicates("pod_id")
    js["bucket"] = js["pod_id"].map(lambda s: zlib.crc32(s.encode()) % 16).astype("int8")
    parts, n_spans, nt = [], 0, 0
    for b in range(16):
        sp = pd.read_parquet(os.path.join(AGG, f"pod_spans_b{b:02d}.parquet"))
        n_spans += len(sp)
        nt = max(nt, int(sp["t_max"].max()) + 1)
        m = sp.merge(js[js["bucket"] == b].drop(columns="bucket"), on="pod_id", how="inner").drop(columns="pod_id")
        parts.append(m)
        log(f"bucket {b}: joined {len(m)}/{len(sp)}")
    df = pd.concat(parts, ignore_index=True); del parts, js
    for c in ["job_type_public", "priority_class", "gpu_spec_public"]:
        df[c] = df[c].astype("category")
    df["delay_h"] = df["schedule_delay_sec"].fillna(0).clip(lower=0) / 3600.0
    df["span_h"] = df["t_max"] - df["t_min"] + 1
    df["h"] = df["gh"] / df["span_h"]
    df["cf_start"] = df["t_min"] - df["delay_h"]
    df["cf_end"] = df["t_max"] + 1 - df["delay_h"]
    df["xpu"] = df["gpu_spec_public"].astype(str).str.startswith("XPU")
    censored = (df["t_min"] <= 0) | (df["cf_start"] < 0)
    stats_cens = {"n_pods": int(len(df)), "n_spans": int(n_spans), "joined_pct_spans": float(len(df) / n_spans * 100),
                  "excluded_censored_pct": float(censored.mean() * 100),
                  "null_delay_gh_pct": float(df.loc[df["schedule_delay_sec"].isna(), "gh"].sum() / df["gh"].sum() * 100),
                  "gh_weighted_mean_delay_h": float((df["delay_h"] * df["gh"]).sum() / df["gh"].sum())}
    df = df[~censored]
    log(f"kept {len(df)} pods ({stats_cens})")

    def wrow(d):
        """vectorised W per busy GPU by (job_type, xpu) for a frame."""
        key = d["job_type_public"].astype(str) + "|" + d["xpu"].astype(int).astype(str)
        table = {f"{k[0]}|{int(k[1])}": v for k, v in wpg.items()}
        return key.map(table).fillna(float(np.nanmean(list(wpg.values())))).to_numpy()

    # ---- D(t) ------------------------------------------------------------------
    A_mw = np.zeros(nt); C_mw = np.zeros(nt); by_type = {}
    for (jt, x), sub in df.groupby(["job_type_public", "xpu"], observed=True):
        w = wpg.get((str(jt), bool(x)), np.nanmean(list(wpg.values()))) / 1e6
        a, c = build_series(sub, nt)
        A_mw += a * w; C_mw += c * w
        pa, pc = by_type.get(str(jt), (np.zeros(nt), np.zeros(nt)))
        by_type[str(jt)] = (pa + a * w, pc + c * w)
    D = A_mw - C_mw
    hrs = np.arange(nt); days = hrs // 24
    good = np.ones(nt, bool); good[:EDGE_DAYS * 24] = False; good[-24:] = False
    post = good & (days >= COVER_DAY + 1)
    summary = {"censoring": stats_cens, "watts_per_busy_gpu": {f"{k[0]}|{'xpu' if k[1] else 'nvidia'}": round(float(v), 1) for k, v in wpg.items()},
               "cover_day": COVER_DAY,
               "battery_full_window": battery_stats(D, A_mw, good, days, f"day {EDGE_DAYS}-{nt//24-1}"),
               "battery_post_coverage": battery_stats(D, A_mw, post, days, f"day {COVER_DAY+1}-{nt//24-1}"),
               "shifted_energy_total_mwh": float((df["h"] * wrow(df) / 1e6
                                                  * np.minimum(df["delay_h"], df["span_h"])).sum())}

    # ---- backlog replay of delay-based deferral (shiftable HP pods) ------------
    sh = df[(df["priority_class"].astype(str) == "HP") & df["job_type_public"].astype(str).isin(pm.SHIFTABLE)].copy()
    sh["w_mw"] = wrow(sh) / 1e6 * sh["h"]
    arr_hour = np.floor(sh["t_min"].to_numpy()).astype(int)
    shift_layer_all = np.zeros(nt + 2)
    add_steps(shift_layer_all, sh["t_min"].to_numpy().astype(float), sh["w_mw"].to_numpy())
    add_steps(shift_layer_all, (sh["t_max"].to_numpy() + 1).astype(float), -sh["w_mw"].to_numpy())
    shift_layer = np.cumsum(shift_layer_all)[:nt]
    replay = {}
    for h in HS:
        elig = sh["delay_h"].to_numpy() >= h
        S = np.zeros(nt)
        np.add.at(S, arr_hour[elig], sh["w_mw"].to_numpy()[elig])          # deferrable arrival power at t
        E_move = np.zeros(nt)                                               # MWh moved by one invocation at t
        np.add.at(E_move, arr_hour[elig], (sh["w_mw"].to_numpy() * np.minimum(h, sh["span_h"].to_numpy()))[elig])
        Sg = np.where(post, S, np.nan)
        replay[f"h{h}"] = {"mean_deferrable_arrival_mw_post": float(np.nanmean(Sg)),
                           "p10_mw_post": float(np.nanquantile(Sg, 0.1)), "k95_mw_post": kval(Sg, h, 0.95),
                           "mean_energy_per_invocation_mwh_post": float(np.nanmean(np.where(post, E_move, np.nan))),
                           "share_of_shift_layer_pct_post": float(np.nanmean(Sg) / np.nanmean(np.where(post, shift_layer, np.nan)) * 100),
                           "mean_deferrable_arrival_mw_full": float(np.nanmean(np.where(good, S, np.nan))),
                           "eligible_pods": int(elig.sum()), "eligible_gh_pct": float(sh["gh"].to_numpy()[elig].sum() / sh["gh"].sum() * 100)}
        pd.DataFrame({"t": hrs, "S_mw": S, "E_move_mwh": E_move}).to_parquet(os.path.join(OUT, f"replay_h{h}.parquet"), index=False)
    summary["backlog_replay"] = replay
    summary["shift_layer_mean_mw_post"] = float(np.nanmean(np.where(post, shift_layer, np.nan)))
    summary["note"] = ("D(t) before the coverage jump is dominated by missing records; use the post-coverage window. "
                       "Backlog replay counts only newly-arriving HP shiftable pods whose observed queueing delay >= h; "
                       "running pods are not paused.")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A3 v2 — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    log(json.dumps({k: summary[k] for k in ["battery_post_coverage", "backlog_replay", "shift_layer_mean_mw_post"]}, indent=1, default=str))

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(hrs[good] / 24, D[good], lw=0.3); axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(COVER_DAY, color="r", ls="--", lw=1, label=f"job-summary coverage jump (day {COVER_DAY})")
    axes[0].set(ylabel="D(t) MW"); axes[0].legend(fontsize=8)
    E = np.cumsum(np.where(good, D, 0.0))
    axes[1].plot(hrs[good] / 24, E[good], lw=0.5, color="tab:orange"); axes[1].axvline(COVER_DAY, color="r", ls="--", lw=1)
    axes[1].set(xlabel="day", ylabel="E(t) MWh")
    fig.suptitle("scheduler shifting (v2 power); pre-jump window is a coverage artefact")
    savefig(fig, "f2_difference_and_energy.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for h, c in zip(HS, ["tab:red", "tab:purple"]):
        S = pd.read_parquet(os.path.join(OUT, f"replay_h{h}.parquet"))["S_mw"].to_numpy()
        ax.plot(hrs[post] / 24, pd.Series(S[post]).rolling(24, min_periods=1).mean(), lw=0.8, color=c,
                label=f"deferrable arrivals, observed delay ≥{h} h (24 h mean)")
    ax.plot(hrs[post] / 24, pd.Series(shift_layer[post]).rolling(24, min_periods=1).mean(), "k", lw=0.8, label="HP shiftable layer (running)")
    ax.set(xlabel="day", ylabel="MW", title="backlog replay: what delay-based deferral can actually move (post-coverage window)")
    ax.legend(fontsize=8)
    savefig(fig, "f6_backlog_replay.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pr, sub in df[df["t_min"] >= (COVER_DAY + 1) * 24].groupby("priority_class", observed=True):
        d_ = sub[sub["delay_h"] > 0]
        if len(d_) < 100:
            continue
        x = np.sort(d_["delay_h"].to_numpy()); wgt = d_.sort_values("delay_h")["gh"].to_numpy()
        ax.plot(x, np.cumsum(wgt) / wgt.sum(), label=f"{pr}")
    ax.set(xscale="log", xlabel="observed queueing delay (h, >0 only)", ylabel="CDF (GPU-hour weighted)",
           title=f"observed queueing delay by priority (day ≥ {COVER_DAY+1})")
    ax.legend()
    savefig(fig, "f5_delay_cdf.png")
    pd.DataFrame({"t": hrs, "actual_mw": A_mw, "cf_mw": C_mw, "d_mw": D, "good": good, "post": post}).to_parquet(
        os.path.join(OUT, "shift_series.parquet"), index=False)
    log("DONE a3_battery_inversion v2")


if __name__ == "__main__":
    main()
