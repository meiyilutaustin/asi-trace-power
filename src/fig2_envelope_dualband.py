#!/usr/bin/env python3
"""E7 (ChatGPT round-3 Q1): redraw the main envelope figure (fig2_envelope) with
TWO distinct uncertainty bands, labelled as different in kind:
  (a) power-model sensitivity  -> the 200-draw Monte-Carlo ensemble (f2_mc_P3),
  (b) time-sampling uncertainty -> the 168 h moving-block bootstrap (f2_boot_P3).
They are NOT merged into a single interval. Base PDF only; the author re-exports
the annotated Picture* version.

Data are read verbatim from the existing figure-data pickle
(runs/2026-09-18_review-B-experiments/figdata_nshift200.pkl), built by
prep_figs.py on the exact E0-E4 substrate. Nothing here recomputes science.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "..", "2026-09-18_review-B-experiments", "figdata_nshift200.pkl")
D = pickle.load(open(PKL, "rb"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
    "legend.fontsize": 6.6, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.dpi": 300,
})

C_ENV = "#0B2F52"        # measured 95% envelope line (dark navy)
C_MODEL = "#C9CBD6"      # power-model sensitivity band (neutral light gray, outer)
C_MODEL_E = "#9A9DB0"    # its edge
C_TIME = "#5B9BD5"       # time-sampling band (blue, inner, nested)
C_TIME_E = "#2E6FA8"     # its edge
C_FIX = "#7A7A7A"
C_INDEP = "#D9600B"      # independent-cluster benchmark (strong orange)
GRIDCOL = "#DFE4EA"
FIX_DASH = (0, (5, 3))
INDEP_DASH = (0, (5, 2, 1, 2))
TICKS = [1, 4, 12, 24]

GRID = np.array(D["GRID"], float)
sel = GRID <= 24
Gx = GRID[sel]

mc = np.asarray(D["f2_mc_P3"])                     # (200 draws, 12 horizons)
mc_p5 = np.nanpercentile(mc, 5, axis=0)[sel]
mc_p95 = np.nanpercentile(mc, 95, axis=0)[sel]
boot = D["f2_boot_P3"]
bt_p5 = np.asarray(boot["p5"])[sel]
bt_p95 = np.asarray(boot["p95"])[sel]
P3 = {a: np.asarray(D["f2_P3"][a])[sel] for a in (0.90, 0.95, 0.99)}
P2_indep = np.asarray(D["f2_P2_indep"])[sel]
avg = D["f2_avg"]
# sanity: the time-sampling band nests inside the power-model band at every horizon
assert np.all(mc_p5 <= bt_p5 + 1e-9) and np.all(bt_p95 <= mc_p95 + 1e-9), "bands not nested"


def dur_axis(ax, ylab):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_xlim(0.92, 25.5)
    ax.set_xlabel("Event duration (hours)")
    ax.set_ylabel(ylab)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDCOL, lw=0.6, zorder=0)
    ax.tick_params(length=3)


def main():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    # (a) power-model sensitivity band -- wide neutral-gray backdrop (outer)
    ax.fill_between(Gx, mc_p5, mc_p95, color=C_MODEL, alpha=0.9, lw=0, zorder=1)
    ax.plot(Gx, mc_p5, color=C_MODEL_E, lw=0.8, zorder=1.5)
    ax.plot(Gx, mc_p95, color=C_MODEL_E, lw=0.8, zorder=1.5)
    # (b) time-sampling band -- blue, narrower, nested inside (inner)
    ax.fill_between(Gx, bt_p5, bt_p95, color=C_TIME, alpha=0.5, lw=0, zorder=2)
    ax.plot(Gx, bt_p5, color=C_TIME_E, lw=0.8, zorder=2.5)
    ax.plot(Gx, bt_p95, color=C_TIME_E, lw=0.8, zorder=2.5)

    # benchmarks
    ax.axhline(avg, color=C_FIX, ls=FIX_DASH, lw=1.4, zorder=4)
    ax.plot(Gx, P2_indep, color=C_INDEP, ls=INDEP_DASH, lw=1.8, marker="s",
            ms=3.4, mec="white", mew=0.4, zorder=6)

    # measured 95% envelope -- the one bold line
    ax.plot(Gx, P3[0.95], color=C_ENV, lw=2.9, zorder=7)

    dur_axis(ax, "Sustained curtailable power (MW)")
    ax.set_ylim(1.4, 3.66)
    ax.annotate("Average eligible load", xy=(1.02, avg - 0.02), fontsize=6.6,
                color=C_FIX, va="top", ha="left")

    handles = [
        Line2D([], [], color=C_ENV, lw=2.9, label="Measured envelope (95% coverage)"),
        Line2D([], [], color=C_INDEP, ls=INDEP_DASH, lw=1.8, marker="s", ms=3.4,
               mec="white", mew=0.4, label="Independent-cluster benchmark"),
        Patch(facecolor=C_MODEL, edgecolor=C_MODEL_E, lw=0.8,
              label="Power-model sensitivity (200-draw ensemble)"),
        Patch(facecolor=C_TIME, alpha=0.5, edgecolor=C_TIME_E, lw=0.8,
              label="Time-sampling (168 h block bootstrap)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="none", facecolor="white", ncol=1, handlelength=1.9,
              labelspacing=0.4, borderaxespad=0.4)

    fig.savefig(os.path.join(HERE, "fig2_envelope_dualband.pdf"),
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(HERE, "fig2_envelope_dualband.png"),
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)

    # numbers for the caption / hand-off
    print("Two bands on fig2_envelope (95% coverage line):")
    print(f"{'T':>3} {'env95':>7} {'model_p5':>9} {'model_p95':>9} {'time_p5':>8} {'time_p95':>8}")
    for i, T in enumerate(Gx):
        print(f"{int(T):>3} {P3[0.95][i]:>7.3f} {mc_p5[i]:>9.3f} {mc_p95[i]:>9.3f} "
              f"{bt_p5[i]:>8.3f} {bt_p95[i]:>8.3f}")
    # relative width comparison at 1/4/24 h
    for T in (1, 4, 24):
        i = int(np.where(Gx == T)[0][0])
        mw = mc_p95[i] - mc_p5[i]; tw = bt_p95[i] - bt_p5[i]
        print(f"  T={T:2d}h: model band {mw:.3f} MW vs time band {tw:.3f} MW  "
              f"(ratio {mw/tw:.2f}x)")
    print("wrote fig2_envelope_dualband.{pdf,png}")


if __name__ == "__main__":
    main()
