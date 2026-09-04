#!/usr/bin/env python
"""Item 8: paper-ready replots of A5 v2 / A6 v2 from saved outputs (local venv).
Usage: replot_v2.py <rerun_dir>   (expects a5/, a6/ subdirs)
Writes into <rerun_dir>/figs_paper/.
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

R = sys.argv[1]   # data/products (public layout) or a run directory with stage subdirs
OUT = os.path.join(R, "figs_paper")
os.makedirs(OUT, exist_ok=True)
HS = [1, 2, 4, 8, 24]
ALPHAS = [0.5, 0.9, 0.95, 0.99]

# ---- A6: K vs duration, legend outside -------------------------------------------
a6 = json.load(open(_find(R, "a6", "summary.json")))
surf = pd.read_parquet(_find(R, "a6", "k_surface.parquet"))
fl = surf[surf.scope == "fleet"]
pick = lambda df, h, a: df[(df.h == h) & (df.alpha == a)].iloc[0]
mean_c = a6["fleet_mean_mw"]["curtail"]
bb = {h: a6["headline_fleet_curtail"].get(f"h{h}_a0.95", {}).get("boot_band_mw_168h") for h in HS}
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for a, c in zip(ALPHAS, ["#bbb", "tab:blue", "tab:green", "tab:red"]):
    ax.plot(HS, [pick(fl, h, a).k_obs for h in HS], "o-", color=c, label=f"observed, α = {a}")
lo = [bb[h]["p5"] if bb[h] else np.nan for h in HS]; hi = [bb[h]["p95"] if bb[h] else np.nan for h in HS]
ax.fill_between(HS, lo, hi, color="tab:green", alpha=0.18, label="α = 0.95: power-MC × 168 h block bootstrap, P5–P95")
ax.plot(HS, [pick(fl, h, .95).k_const for h in HS], "s--", color="k", label="constant share calibrated to the mean")
ax.plot(HS, [pick(fl, h, .95).k_shuf for h in HS], "^:", color="tab:purple", label="time-shuffled (persistence removed)")
ax.plot(HS, [pick(fl, h, .95).k_indep_v2 for h in HS], "v:", color="tab:orange", label="clusters independent (detrended shift)")
ax.axhline(mean_c, color="gray", lw=0.8, ls="-.", label=f"mean eligible-workload availability {mean_c:.2f} MW")
ax.set_xscale("log"); ax.set_xticks(HS, [str(h) for h in HS])
ax.set(xlabel="event duration h (hours)", ylabel="K(α, h)  [MW, GPU side]",
       title=f"eligible-workload availability: duration–reliability surface (N = {a6['n_clusters']} logical clusters)")
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.savefig(os.path.join(OUT, "F_k_surface.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A6: portfolio ----------------------------------------------------------------
rnd = surf[surf.scope == "random"]
N = a6["n_clusters"]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for h, c in zip([1, 4, 24], ["tab:blue", "tab:green", "tab:red"]):
    sub = rnd[(rnd.h == h) & (rnd.alpha == .95)].sort_values("n")
    ax.plot(sub.n, sub.firm_obs, "o-", color=c, label=f"observed, h = {h}")
sub = rnd[(rnd.h == 4) & (rnd.alpha == .95)].sort_values("n")
ax.plot(sub.n, sub.firm_indep_v2, "v:", color="tab:green", label="independence null (detrended), h = 4")
ax.plot(sub.n, sub.firm_const, "s--", color="k", label="constant share, h = 4")
bf = a6["boot_firmness_nested_h4_a95"]
ns = np.arange(1, N + 1)
ax.fill_between(ns, [bf[str(n)]["p5"] for n in ns], [bf[str(n)]["p95"] for n in ns], color="tab:green", alpha=0.15, label="nested top-n, h = 4, bootstrap P5–P95")
ax.set(xlabel="portfolio size n (logical clusters, all C(N,n) portfolios)", ylabel="K(0.95, h) / mean", ylim=(0, 1.0),
       title="aggregation of eligible-workload availability across logical clusters")
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(os.path.join(OUT, "F_portfolio.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A5: average-day envelope, idle-retained as the main boundary -----------------
env = pd.read_parquet(_find(R, "a5", "envelope_hourly.parquet"))
a5 = json.load(open(_find(R, "a5", "summary.json")))
good = env["good"].to_numpy().astype(bool)
hod = env["t"].to_numpy() % 24
avg = lambda col: np.array([env.loc[good & (hod == h), col].mean() for h in range(24)])
xs = np.arange(24)
fig, ax = plt.subplots(figsize=(7.8, 4.8))
stack = np.zeros(24)
for L, c in [("floor", "#777"), ("shift", "tab:orange"), ("curtail", "tab:green"), ("standby", "tab:blue")]:
    v = avg(L)
    ax.fill_between(xs, stack, stack + v, color=c, alpha=0.85, label={"floor": "must-run floor", "shift": "shiftable (HP training / offline)",
                                                                       "curtail": "eligible-attributed workload (semantic upper bound)", "standby": "standby"}[L])
    stack = stack + v
ax.plot(xs, stack - avg("curtail_idle_retained"), "k-", lw=1.4, label="after preemption, GPU idle retained (default)")
ax.plot(xs, stack - avg("curtail_attributed"), "k:", lw=1.0, label="after preemption, attributed boundary (upper bound)")
b = a5["mc_bands"]["curtail_idle_retained_pct"]
ax.set(xlabel="hour of day", ylabel="GPU-side workload power (MW)",
       title=f"average-day envelope; eligible-workload availability = {b['p50']:.1f}% of workload (MC P5–P95 {b['p5']:.1f}–{b['p95']:.1f}%)")
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(os.path.join(OUT, "F_envelope_avgday.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A5: eligibility x boundary bars ----------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.2))
E = ["conservative", "central", "liberal"]; B = ["attributed", "idle_retained"]
x = np.arange(len(E))
for j, bname in enumerate(B):
    vals = [a5["by_eligibility"][e]["curtail_mw"][bname]["mean"] for e in E]
    ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label={"attributed": "attributed (whole pod power)", "idle_retained": "idle retained (default)"}[bname],
           color=["tab:gray", "tab:green"][j])
    for xi, v in zip(x + (j - 0.5) * 0.35, vals):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, E)
ax.set(ylabel="mean eligible-workload availability (MW, GPU side)", title="eligibility mapping × electrical boundary")
ax.legend(fontsize=8)
fig.savefig(os.path.join(OUT, "F_eligibility_boundary.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A5: six-month semantic layers, with electrical boundary explicit -----------
layer_colors = {"floor": "#666", "shift": "tab:orange", "curtail": "tab:green", "standby": "tab:blue"}
layer_labels = {"floor": "must-run floor",
                "shift": "shiftable semantic pool (HP training / offline)",
                "curtail": "eligible-attributed workload (semantic upper bound)",
                "standby": "standby"}
fig, ax = plt.subplots(figsize=(12.5, 4.8))
tday = env["t"].to_numpy() / 24
ordered_layers = ["floor", "shift", "curtail", "standby"]
stack_vals = [np.where(good, env[L].to_numpy(), 0) for L in ordered_layers]
ax.stackplot(tday, *stack_vals, labels=[layer_labels[L] for L in ordered_layers],
             colors=[layer_colors[L] for L in ordered_layers], alpha=0.85)
ax.plot(tday, np.where(good, env["workload"] - env["curtail_idle_retained"], np.nan),
        "k--", lw=0.7, label="after preemption: eligible-workload availability (idle retained)")
ax.set(xlabel="day", ylabel="GPU-side workload power (MW)", title="workload semantic layers over six months")
ax.legend(loc="upper left", fontsize=7.5)
fig.savefig(os.path.join(OUT, "F_envelope_layers.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A4: evolution of the same semantic layers ----------------------------------
evo = pd.DataFrame.from_dict(a5["a4_monthly_layer_share_pct"], orient="index")
evo.index = evo.index.astype(int)
evo = evo.sort_index()
fig, ax = plt.subplots(figsize=(8.2, 4.5))
evo.plot(kind="bar", stacked=True, ax=ax, width=0.85,
         color=[layer_colors.get(c, "#999") for c in evo.columns])
ax.set(xlabel="30-day block start", ylabel="share of workload power (%)", title="evolution of workload semantic layers")
ax.legend([layer_labels.get(c, c) for c in evo.columns], fontsize=7.5, loc="lower right")
fig.savefig(os.path.join(OUT, "F_evolution.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- A6: eligibility-only firmness grid -----------------------------------------
M = np.array([[pick(fl, h, a).k_obs / mean_c for a in ALPHAS] for h in HS])
fig, ax = plt.subplots(figsize=(6.2, 4.5))
im = ax.imshow(M, vmin=0, vmax=1.0, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(ALPHAS)), [str(a) for a in ALPHAS]); ax.set_yticks(range(len(HS)), [str(h) for h in HS])
ax.set(xlabel="reliability α", ylabel="duration h (hours)", title="eligible-workload availability firmness")
for i in range(len(HS)):
    for j in range(len(ALPHAS)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.7 else "k", fontsize=8)
fig.colorbar(im, ax=ax, label="K / mean eligible-workload availability")
fig.savefig(os.path.join(OUT, "F_firmness_heatmap.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", os.listdir(OUT))
