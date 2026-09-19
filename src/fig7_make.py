#!/usr/bin/env python3
"""Rebuild fig7_decisions with panel (a) = availability ratio vs portfolio size
(方案 B). Panels (b) and (c) are reproduced byte-for-byte from the restructure
base (same code, same cached figdata pickle) — only panel (a) changes.

Panel (a) reads runs/2026-09-18_fig7a-portfolio/portfolio_scan.json:
  observed curve  = subset-mean 4-hour availability ratio (envelope / mean
                    eligible power), n = 1..13 pooled clusters;
  benchmark       = independent-cluster (detrended, decorrelated) planner;
  ribbon          = nested top-n 168 h moving-block bootstrap P5-P95 (from a6),
                    shown as a RELATIVE band around the plotted subset-mean line
                    (nested p5/p95 divided by nested base, times the subset-mean),
                    so the shading conveys bootstrap magnitude while staying
                    centred on the estimator actually drawn. See README.
Neutral title per style/rules.md rule 13: names the quantity only.
"""
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SC = os.environ.get("FIGCACHE", os.path.join(tempfile.gettempdir(), "restructure_figdata.pkl"))
D = pickle.load(open(SC, "rb"))
SCAN = json.loads((HERE / "portfolio_scan.json").read_text())

# ---- shared style (verbatim from restructure make_figs.py) ------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 300,
})
TAB10 = plt.get_cmap("tab10").colors
GRIDCOL = "#DFE4EA"
TITLECOL = "#1a1a1a"
C_MEAS = "#0B3D91"; C_RIBBON = "#C4CCD6"; C_INDEP = "#D1495B"
FIX_DASH = (0, (5, 3)); INDEP_DASH = (0, (5, 2, 1, 2))
LW_MAIN, LW_SEC, LW_REF = 2.8, 2.2, 1.3
F7C = {"envelope": TAB10[0], "fixed": TAB10[7], "noflex": TAB10[3], "full": TAB10[6],
       "T1": TAB10[1], "T4": TAB10[2], "T24": TAB10[4],
       "c90": TAB10[5], "c95": TAB10[8], "c99": TAB10[9]}

OUT = HERE

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.6, 2.7))

# ---------------------------------------------------------------- panel (a) NEW
h4 = SCAN["by_horizon"]["h4"]
ns = np.array(sorted(int(k) for k in h4), dtype=float)
obs = np.array([h4[str(int(n))]["firm_obs"] for n in ns])
ind = np.array([h4[str(int(n))]["firm_indep"] for n in ns])
rib = SCAN["observed_ribbon_h4_nested_top_n_168h_block_bootstrap"]
base = np.array([rib[str(int(n))]["base"] for n in ns])
p5 = np.array([rib[str(int(n))]["p5"] for n in ns])
p95 = np.array([rib[str(int(n))]["p95"] for n in ns])
# relative band centred on the subset-mean observed line
lo = obs * (p5 / base)
hi = obs * (p95 / base)

axa.axhline(1.0, color="#8A8A8A", ls=(0, (1, 2)), lw=1.1, zorder=1)
axa.fill_between(ns, lo, hi, color=C_RIBBON, alpha=0.7, lw=0, zorder=2)
axa.plot(ns, ind, color=C_INDEP, ls=INDEP_DASH, lw=LW_SEC, marker="^", ms=5.2,
         mec="white", mew=0.6, zorder=4, label="Independent-cluster benchmark")
axa.plot(ns, obs, color=C_MEAS, lw=LW_MAIN, marker="o", ms=5.4,
         mec="white", mew=0.6, zorder=5, label="Observed (pooled clusters)")
axa.set_xlim(0.6, 13.4)
axa.set_xticks([1, 4, 7, 10, 13])
axa.set_ylim(0, 1.06)
axa.set_xlabel("Number of clusters pooled")
axa.set_ylabel("4-hour availability ratio\n(envelope / mean eligible power)")
axa.set_title("Four-hour availability ratio versus portfolio size",
              color=TITLECOL, fontweight="bold", pad=6, fontsize=7.6, wrap=True)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)
axa.grid(color=GRIDCOL, lw=0.6, zorder=0)
axa.tick_params(length=3)
handles = [
    Line2D([], [], color=C_MEAS, lw=LW_MAIN, marker="o", ms=5.4, mec="white", mew=0.6,
           label="Observed (pooled clusters)"),
    Line2D([], [], color=C_INDEP, ls=INDEP_DASH, lw=LW_SEC, marker="^", ms=5.2, mec="white", mew=0.6,
           label="Independent-cluster benchmark"),
    Line2D([], [], color="#8A8A8A", ls=(0, (1, 2)), lw=1.1, label="Mean eligible power"),
]
axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.99),
           frameon=True, framealpha=0.95, edgecolor="#B0B0B0", fancybox=False,
           handlelength=2.0, borderpad=0.5, labelspacing=0.35, fontsize=7.6)

# ------------------------------------------------ panels (b),(c): base verbatim
cov = D["f7_cov"] * 100
for T, key in ((1, "T1"), (4, "T4"), (24, "T24")):
    y = D["f7_grid"][T]
    if T == 24:
        # 24-hour curve: solid where the tail is well supported (coverage <= 95%),
        # dashed beyond, where the running-min tail rests on only a handful of
        # independent trough events (E4: 8 events at 95%, 2 at 99%). The dashed
        # segment marks statistically thin support, per reviewer request.
        i = int(np.searchsorted(cov, 95.0))
        axb.plot(cov[:i + 1], y[:i + 1], color=F7C[key], lw=LW_MAIN, label="24-hour event")
        axb.plot(cov[i:], y[i:], color=F7C[key], lw=LW_MAIN, ls=(0, (4, 2)))
    else:
        axb.plot(cov, y, color=F7C[key], lw=LW_MAIN, label=f"{T}-hour event")
axb.set_xlabel("Required coverage (%)")
axb.set_ylabel("Sustained curtailable power (MW)")
axb.set_xlim(80, 99.9)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.grid(color=GRIDCOL, lw=0.6, zorder=0); axb.tick_params(length=3)

FR = D["f7_frac"] * 100
for a, key in ((0.90, "c90"), (0.95, "c95"), (0.99, "c99")):
    axc.plot(FR, D["f7_oper"][a], color=F7C[key], lw=LW_MAIN, label=f"{int(a*100)}% coverage")
axc.set_xlabel("High-priority training enrolled (%)")
axc.set_ylabel("4-hour sustained curtailable power (MW)")
axc.set_xlim(0, 100)
for s in ("top", "right"):
    axc.spines[s].set_visible(False)
axc.grid(color=GRIDCOL, lw=0.6, zorder=0); axc.tick_params(length=3)

fig.tight_layout(w_pad=2.4)
fig.savefig(f"{OUT}/fig7_decisions.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{OUT}/fig7_decisions.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
plt.close(fig)
print("wrote fig7_decisions.pdf/.png ->", OUT)
