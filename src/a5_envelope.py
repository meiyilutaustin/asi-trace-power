#!/usr/bin/env python
"""A5 (v2, revision 2): hourly flexibility envelope.

Layers per hour (fleet and per cluster), from the A1 aggregate table with the
shared power model (src/powermodel.py; online-inference floor included):
  floor    must-run (HP online / other / unknown; plus LP rows not eligible)
  curtail  eligible-for-preemption rows (mapping = --eligibility / cfg)
  shift    HP training + offline_inference, scaled by the observed queueing
           delay share s_tau(type, priority) = P(delay >= tau) for tau=1,4,24 h
  standby  Standby-state pods (idle power held in reserve)

Curtailment is reported on THREE electrical boundaries :
  attributed    = whole pod GPU power (old headline, upper bound)
  idle_retained = only the active part (the reclaimed GPU keeps idling; default)
  node_sleep    = attributed (node power-gated)
plus an optional host add-on (dynamic CPU power of the evicted pods).
Facility conversion: average PUE (cfg.pue.base) and marginal (cfg.pue_marginal).
Shift products are horizon-specific : the "DR" headline = curtail + shift_ge1h
and is stated for the 1 h product only.
MC bands (formerly a5_uncertainty.py) are computed here for the headline metrics.
Outputs to $OUT_DIR: envelope_hourly.parquet (fleet), envelope_cluster_hourly.parquet
(long, per cluster), summary.json/.md, figures.
"""
import argparse
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

DATA = os.environ.get("DATA_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "data"))
AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a5_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(
    os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
TAUS = [1, 4, 24]
LAYERS = ["floor", "shift", "curtail", "standby"]


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def tolerance():
    js = pd.read_parquet(
        os.path.join(DATA, "asi_opensource_job_execution_summary"),
        columns=["gpu_request", "duration_hours", "schedule_delay_sec",
                 "job_type_public", "priority_class"])
    js["w"] = (js["gpu_request"].fillna(0).clip(lower=0)
               * js["duration_hours"].fillna(0).clip(lower=0))
    js["dh"] = js["schedule_delay_sec"].fillna(0).clip(lower=0) / 3600
    out = {}
    for (jt, pr), g in js.groupby(["job_type_public", "priority_class"], observed=True):
        tot = g["w"].sum()
        if tot > 0:
            out[(jt, pr)] = {t: float(g.loc[g["dh"] >= t, "w"].sum() / tot) for t in TAUS}
    return out


def series(nt, tpod, mask, vals):
    a = np.zeros(nt)
    np.add.at(a, tpod[mask], vals[mask])
    return a


def layer_series(pod, cfg, p, layer, spec_codes, spec_uniq, tpod, nt, s_tau, host_dyn):
    """All fleet series (MW, GPU side unless noted) for one param draw."""
    parts = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)
    tot = parts["total"] / 1e6
    out = {L: series(nt, tpod, layer == L, tot) for L in LAYERS}
    cur = layer == "curtail"
    out["curtail_attributed"] = out["curtail"]
    out["curtail_idle_retained"] = series(nt, tpod, cur, parts["active"] / 1e6)
    out["curtail_node_sleep"] = out["curtail_attributed"]
    out["curtail_host"] = series(nt, tpod, cur, host_dyn / 1e6) if host_dyn is not None else np.zeros(nt)
    sh = layer == "shift"
    for tau in TAUS:
        out[f"shift_ge{tau}h"] = series(nt, tpod, sh, tot * s_tau[tau])
    out["workload"] = sum(out[L] for L in LAYERS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligibility", default=None, choices=list(pm.ELIGIBILITY))
    ap.add_argument("--mc", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    elig = args.eligibility or cfg.get("eligibility", "central")
    boundary = cfg.get("curtail_boundary", "idle_retained")
    pue_avg, pue_marg = cfg["pue"]["base"], cfg.get("pue_marginal", 1.0)
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    tol = tolerance()
    keys = list(zip(pod["job_type_public"].astype(str), pod["priority_class"].astype(str)))
    s_tau = {tau: np.array([tol.get(k, {tau: 0.0})[tau] for k in keys]) for tau in TAUS}
    rng = np.random.default_rng(int(cfg.get("mc_seed", 0)) + 5)
    p0 = pm.draw_params(cfg, rng, specs, base=True)
    host_dyn = pm.host_energy_rows(pod, p0)

    # ---- base run for every eligibility mapping ------------------------------
    per_elig = {}
    for e in pm.ELIGIBILITY:
        layer = pm.assign_layer(pod, e)
        per_elig[e] = layer_series(pod, cfg, p0, layer, spec_codes, spec_uniq, tpod, nt, s_tau, host_dyn)
    env = per_elig[elig]
    good = env["workload"] > 0
    layer = pm.assign_layer(pod, elig)

    def stats(o):
        w = o["workload"][good].mean()
        c = o[f"curtail_{boundary}"]
        return {"layer_share_pct": {L: float(o[L][good].mean() / w * 100) for L in LAYERS},
                "curtail_mw": {b: {"mean": float(o[f"curtail_{b}"][good].mean()),
                                   "p10": float(np.quantile(o[f"curtail_{b}"][good], 0.1))}
                               for b in pm.BOUNDARIES},
                "curtail_host_mw_mean": float(o["curtail_host"][good].mean()),
                "shift_mw_ge_tau": {f"{t}h": {"mean": float(o[f"shift_ge{t}h"][good].mean()),
                                              "p10": float(np.quantile(o[f"shift_ge{t}h"][good], 0.1))}
                                    for t in TAUS},
                "dr1h_mw_mean": float((c + o["shift_ge1h"])[good].mean()),
                "dr1h_pct_of_workload": float((c + o["shift_ge1h"])[good].mean() / w * 100),
                "curtail_pct_of_workload": float(c[good].mean() / w * 100),
                "workload_mw_mean": float(w)}

    summary = {"eligibility": elig, "curtail_boundary": boundary,
               "pue": {"average": pue_avg, "marginal": pue_marg},
               "central": stats(env),
               "by_eligibility": {e: stats(per_elig[e]) for e in pm.ELIGIBILITY},
               "tolerance_table": {f"{k[0]}|{k[1]}": v for k, v in tol.items()}}

    # ---- per-cluster series (base params, chosen eligibility) ---------------
    cl_codes, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
    parts = pm.row_energy(pod, cfg, p0, spec_codes, spec_uniq)
    rows = []
    for ci, cname in enumerate(cl_uniq):
        m = cl_codes == ci
        rec = {"cluster_id": str(cname)}
        d = {}
        for L in LAYERS:
            d[L] = series(nt, tpod, m & (layer == L), parts["total"] / 1e6)
        d["curtail_idle_retained"] = series(nt, tpod, m & (layer == "curtail"), parts["active"] / 1e6)
        d["curtail_host"] = series(nt, tpod, m & (layer == "curtail"), host_dyn / 1e6) if host_dyn is not None else np.zeros(nt)
        for tau in TAUS:
            d[f"shift_ge{tau}h"] = series(nt, tpod, m & (layer == "shift"), parts["total"] / 1e6 * s_tau[tau])
        df = pd.DataFrame({"t": np.arange(nt), **d})
        df.insert(0, "cluster_id", str(cname))
        df["curtail_attributed"] = df["curtail"]
        rows.append(df)
    clus = pd.concat(rows, ignore_index=True)
    clus["good"] = clus["t"].map(pd.Series(good))
    clus.to_parquet(os.path.join(OUT, "envelope_cluster_hourly.parquet"), index=False)
    # coincidence of the default-boundary curtailable layer, all clusters with any
    cw = clus.pivot(index="t", columns="cluster_id", values=f"curtail_{boundary}")
    cw = cw.loc[good, cw.mean() > 0.0]
    corr = cw.corr()
    iu = np.triu_indices(len(corr), 1)
    summary["cluster_coincidence"] = {
        "n_clusters_with_curtail": int(cw.shape[1]),
        "corr_median_all": float(np.median(corr.values[iu])),
        "corr_mean_all": float(np.mean(corr.values[iu])),
        "corr_median_top8": float(np.median(corr.loc[cw.mean().nlargest(8).index,
                                                     cw.mean().nlargest(8).index].values[np.triu_indices(8, 1)]))
        if cw.shape[1] >= 8 else None}

    # ---- A4 monthly evolution ------------------------------------------------
    day = np.arange(nt) // 24
    ev = pd.DataFrame({"blk": day // 30, **{L: env[L] for L in LAYERS}})[good]
    ev = ev.groupby("blk").sum()
    ev = ev.div(ev.sum(axis=1), axis=0) * 100
    summary["a4_monthly_layer_share_pct"] = {
        str(int(k) * 30): {c: round(float(v), 2) for c, v in row.items()} for k, row in ev.iterrows()}
    sb = pod[(pod["state_public"] == "Standby") & (pod["gpu_hours"] > 0)]
    summary["standby_first_day"] = int(sb["day"].astype(int).min()) if len(sb) else None

    # ---- MC bands for the headline metrics (replaces a5_uncertainty.py) -----
    ns = args.mc if args.mc is not None else int(cfg.get("mc_samples", 300))
    keys_mc = ["workload_mw", "curtail_attributed_mw", "curtail_idle_retained_mw",
               "curtail_idle_retained_p10_mw", "curtail_idle_retained_pct",
               "dr1h_pct", "floor_pct", "shift1_mw"]
    mc = {k: np.zeros(ns) for k in keys_mc}
    avgday = {L: np.zeros((ns, 24)) for L in ["floor", "shift", "curtail"]}
    hod = np.arange(nt) % 24
    for i in range(ns):
        p = pm.draw_params(cfg, rng, specs, base=(i == 0))
        o = layer_series(pod, cfg, p, layer, spec_codes, spec_uniq, tpod, nt, s_tau, host_dyn)
        w = o["workload"][good].mean()
        cir = o["curtail_idle_retained"]
        mc["workload_mw"][i] = w
        mc["curtail_attributed_mw"][i] = o["curtail_attributed"][good].mean()
        mc["curtail_idle_retained_mw"][i] = cir[good].mean()
        mc["curtail_idle_retained_p10_mw"][i] = np.quantile(cir[good], 0.1)
        mc["curtail_idle_retained_pct"][i] = cir[good].mean() / w * 100
        mc["dr1h_pct"][i] = (cir + o["shift_ge1h"])[good].mean() / w * 100
        mc["floor_pct"][i] = o["floor"][good].mean() / w * 100
        mc["shift1_mw"][i] = o["shift_ge1h"][good].mean()
        for L in avgday:
            for h in range(24):
                sel = good & (hod == h)
                avgday[L][i, h] = o[L][sel].mean()
        if (i + 1) % 50 == 0:
            log(f"MC {i+1}/{ns}")
    band = lambda a: {"p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                      "p95": float(np.percentile(a, 95)), "base": float(a[0])}
    summary["mc_bands"] = {k: band(v) for k, v in mc.items()}
    summary["mc_samples"] = ns

    # ---- outputs ---------------------------------------------------------------
    fleet = pd.DataFrame({"t": np.arange(nt), "good": good, **{k: v for k, v in env.items()}})
    fleet.to_parquet(os.path.join(OUT, "envelope_hourly.parquet"), index=False)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A5 envelope v2 — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    log(json.dumps({k: summary["central"][k] for k in ["layer_share_pct", "curtail_mw", "dr1h_pct_of_workload"]}, indent=1))

    hrs = np.arange(nt)
    cols = {"floor": "#666", "shift": "tab:orange", "curtail": "tab:green", "standby": "tab:blue"}
    layer_labels = {"floor": "must-run floor",
                    "shift": "shiftable semantic pool (HP training / offline)",
                    "curtail": "eligible-attributed workload (semantic upper bound)",
                    "standby": "standby"}
    fig, ax = plt.subplots(figsize=(13, 5))
    em = {L: np.where(good, env[L], 0) for L in LAYERS}
    ax.stackplot(hrs / 24, *[em[L] for L in LAYERS], labels=[layer_labels[L] for L in LAYERS],
                 colors=[cols[L] for L in LAYERS], alpha=0.85)
    ax.plot(hrs / 24, np.where(good, env["workload"] - env[f"curtail_{boundary}"], np.nan), "k--", lw=0.6,
            label=f"after preemption: eligible-workload availability ({boundary})")
    ax.set(xlabel="day", ylabel="GPU-side workload MW", title=f"workload semantic layers, 6 months ({elig} eligibility)")
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, "f1_layers_series.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(24)
    stack = np.zeros(24)
    for L in ["floor", "shift", "curtail"]:
        p50 = np.percentile(avgday[L], 50, axis=0)
        ax.fill_between(xs, stack, stack + p50, color=cols[L], alpha=0.85, label=L)
        stack = stack + p50
    ir = np.array([env["curtail_idle_retained"][good & (hod == h)].mean() for h in range(24)])
    ax.plot(xs, stack - ir, "k--", lw=1.2, label="min feasible: preempt LP (idle retained)")
    ax.plot(xs, stack - np.array([env["curtail_attributed"][good & (hod == h)].mean() for h in range(24)]),
            "k:", lw=1.0, label="min feasible: attributed boundary")
    b = summary["mc_bands"]
    ax.set(xlabel="hour of day", ylabel="MW (P50 layers)",
           title=f"average-day envelope; curtailable (idle-retained) = {b['curtail_idle_retained_pct']['p50']:.1f}% "
                 f"(P5–P95 {b['curtail_idle_retained_pct']['p5']:.1f}–{b['curtail_idle_retained_pct']['p95']:.1f}%)")
    ax.legend(fontsize=8)
    savefig(fig, "f2_avgday_envelope.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for bname, c in zip(pm.BOUNDARIES, ["tab:gray", "tab:green", "tab:olive"]):
        v = np.sort(env[f"curtail_{bname}"][good])[::-1]
        ax.plot(np.linspace(0, 100, len(v)), v, color=c, label=f"curtailable: {bname}")
    for tau, c in zip(TAUS, ["tab:red", "tab:purple", "tab:brown"]):
        v = np.sort(env[f"shift_ge{tau}h"][good])[::-1]
        ax.plot(np.linspace(0, 100, len(v)), v, color=c, ls="--", label=f"shiftable ≥{tau} h (observed delay)")
    ax.set(xlabel="% of hours", ylabel="MW", title="flexible capacity duration curves")
    ax.legend(fontsize=8)
    savefig(fig, "f3_duration_curves.png")

    top = cw.mean().nlargest(min(8, cw.shape[1])).index
    coin = cw[top].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(coin.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(top)), [str(x)[:8] for x in top], rotation=90)
    ax.set_yticks(range(len(top)), [str(x)[:8] for x in top])
    fig.colorbar(im, label=f"corr of curtailable MW ({boundary})")
    ax.set(title="multi-cluster coincidence (top-8 clusters)")
    savefig(fig, "f4_coincidence.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ev.plot(kind="bar", stacked=True, ax=ax, width=0.85, color=[cols.get(c, "#999") for c in ev.columns])
    ax.set(xlabel="30-day block", ylabel="share of workload power (%)", title="A4: layer shares over six months")
    ax.legend([layer_labels.get(c, c) for c in ev.columns], fontsize=8)
    savefig(fig, "f5_a4_evolution.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pm.ELIGIBILITY))
    for j, bname in enumerate(pm.BOUNDARIES):
        ax.bar(x + (j - 1) * 0.25, [summary["by_eligibility"][e]["curtail_mw"][bname]["mean"] for e in pm.ELIGIBILITY],
               0.25, label=bname)
    ax.set_xticks(x, list(pm.ELIGIBILITY))
    ax.set(ylabel="mean curtailable MW (GPU side)", title="eligibility mapping × electrical boundary")
    ax.legend(fontsize=8)
    savefig(fig, "f6_eligibility_boundary.png")
    log("DONE a5_envelope v2")


if __name__ == "__main__":
    main()
