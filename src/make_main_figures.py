#!/usr/bin/env python3
"""Render the 7 restructured main-paper figures to the reviewer redesign in
`../paper-gpu-power/reviews/20260915chatgpt.md`.

Design objective (reviewer): every figure answers ONE question a reader can grasp
without the body text. Concretely this pass follows the reviewer rather than the
earlier RESTRUCTURE.md style rules where the two conflict:
  - every panel carries a short NEUTRAL title that names the quantity plotted
    (e.g. "Coincidence factor versus event duration") -- never a takeaway, claim,
    or observed phenomenon; the interpretation lives in the caption and body.
    This follows `style/rules.md` rule 13, which OVERRIDES the reviewer note
    (reviews/20260915chatgpt.md) where that note asked for message/takeaway
    titles. (RESTRUCTURE had forbidden on-figure text entirely; we keep a title
    but hold it to a neutral description.)
  - plain English everywhere (no LP / HP / pct / P2 / P3 / "own mean");
  - axes state quantity + units; legends sit in whitespace, never over data;
  - no spaghetti of raw draws in the main panels -> bootstrap ribbons instead;
  - Fig. 2 is a single panel (old panel b removed); Fig. 3 is three real funnels
    with widths proportional to MW; Fig. 7 is three stakeholder decision panels.
Numbers are read from figdata.pkl (built by prep_figs.py on the exact E0-E4
substrate); nothing here recomputes science. See CHANGELOG_figures.md for the
full list of deviations, aggregations and removed elements.
"""
import os, pickle, tempfile, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter, LogLocator, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D

SC = os.environ.get("FIGCACHE", os.path.join(tempfile.gettempdir(), "restructure_figdata.pkl"))
OUT = os.environ.get("FIGOUT", os.path.join(tempfile.gettempdir(), "restructure_figs"))
D = pickle.load(open(SC, "rb"))
os.makedirs(OUT, exist_ok=True)

# ---- shared style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 300,
})

# Colorblind-safe semantic palette (reviewer rule 6). One entity = one colour.
C_MEAS   = "#2166AC"   # measured flexibility envelope (blue)
C_RIBBON = "#9FC3E0"   # uncertainty around the measured estimate (light blue)
C_FIX    = "#7A7A7A"   # fixed-share / average benchmark (neutral gray, dashed)
C_INDEP  = "#E08214"   # independent-cluster benchmark (muted orange, dash-dot)
C_PERS   = "#E08214"   # persistence effect (orange)
C_COINC  = "#2166AC"   # coincidence effect / final sustained envelope (blue)
C_HEL    = "#238B8B"   # external operator validation (teal)
C_GOLD   = "#C79A3E"   # expanded workload scope / added participation (muted gold)
BLUE3    = {0.90: "#8CB3D9", 0.95: "#2166AC", 0.99: "#0B2F52"}   # coverage shades
GRIDCOL  = "#DFE4EA"
TITLECOL = "#1a1a1a"

FIX_DASH   = (0, (5, 3))
INDEP_DASH = (0, (5, 2, 1, 2))
LW_MAIN, LW_SEC, LW_REF = 2.2, 1.6, 1.3

GRID = np.array(D["GRID"], float)
XMAX = 24
sel = GRID <= XMAX
Gx = GRID[sel]
TICKS = [1, 4, 12, 24]
idx = lambda t: list(D["GRID"]).index(t)


def wrap_title(ax, s, width=34, loc="center", pad=6, size=None):
    kw = {} if size is None else {"fontsize": size}
    ax.set_title("\n".join(textwrap.wrap(s, width)), color=TITLECOL,
                 fontweight="bold", loc=loc, pad=pad, **kw)


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


def panel_tag(ax, s, dx=-0.09, dy=1.16):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="right", color="#333")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print("wrote", name)


# ============================================================ FIG 1 ===========
# Conceptual schematic. Normalized so the fixed-share estimate = 1.0; the blue
# curve and band are the REAL measured envelope divided by the average eligible
# load, which is a monotone illustration of the concept without arbitrary MW.
def fig1():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    avg = D["f2_avg"]
    meas = D["f2_P3"][0.95][sel] / avg
    lo = D["f2_P3"][0.99][sel] / avg
    hi = D["f2_P3"][0.90][sel] / avg
    ax.fill_between(Gx, lo, hi, color=C_RIBBON, alpha=0.55, lw=0, zorder=2,
                    label="Flexibility envelope\n(across coverage conditions)")
    ax.axhline(1.0, color=C_FIX, ls=FIX_DASH, lw=1.8, zorder=3,
               label="Fixed-share estimate")
    ax.plot(Gx, meas, color=C_MEAS, lw=2.8, zorder=4,
            label="Sustained curtailable power")
    ax.annotate("Sustained availability falls\nas events get longer",
                xy=(12, meas[idx(12)]), xytext=(4.2, 0.50),
                fontsize=7.2, color="#333", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_xlim(0.92, 25.5)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("Event duration (hours)")
    ax.set_ylabel("Normalized curtailable power")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDCOL, lw=0.6, zorder=0)
    ax.tick_params(length=3)
    wrap_title(ax, "Normalized curtailable power versus event duration, by coverage level",
               width=42)
    ax.legend(loc="lower left", frameon=False, handlelength=1.9,
              labelspacing=0.7, borderpad=0.3, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig1_concept")


# ============================================================ FIG 2 ===========
# Single panel, short. No on-figure title, no descriptive note. tab10 palette so
# every line is a distinct hue; boxed legend on the right, ordered top->bottom to
# match the on-plot vertical order of the lines (fixed > independent > 90 > 95 >
# 99 coverage). One bootstrap ribbon (around the 95% curve).
# tab10-derived colours: fixed=gray, independent=orange, 90/95/99=blue/green/purple
F2_FIX   = "#d62728"                       # fixed-share benchmark (tab10 red, dashed)
F2_INDEP = "#ff7f0e"                       # independent-cluster benchmark (orange)
F2_COV   = {0.90: "#1f77b4",               # measured 90% coverage (blue)
            0.95: "#2ca02c",               # measured 95% coverage (green, primary)
            0.99: "#9467bd"}               # measured 99% coverage (purple)
F2_RIBBON = "#cfcfcf"                      # sampling-uncertainty band (light gray)
def fig2():
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    b = D["f2_boot_P3"]
    ax.fill_between(Gx, b["p5"][sel], b["p95"][sel], color=F2_RIBBON, alpha=0.7,
                    lw=0, zorder=2)
    ax.axhline(D["f2_avg"], color=F2_FIX, ls=FIX_DASH, lw=1.6, zorder=3)
    ax.plot(Gx, D["f2_P2_indep"][sel], color=F2_INDEP, ls=INDEP_DASH, lw=1.8,
            marker="s", ms=3.5, zorder=4)
    # 90% and 99% secondary, 95% primary (thick)
    ax.plot(Gx, D["f2_P3"][0.90][sel], color=F2_COV[0.90], lw=1.9, zorder=5)
    ax.plot(Gx, D["f2_P3"][0.99][sel], color=F2_COV[0.99], lw=1.9, zorder=5)
    ax.plot(Gx, D["f2_P3"][0.95][sel], color=F2_COV[0.95], lw=2.8, zorder=6)
    dur_axis(ax, "Sustained curtailable power (MW)")
    ax.set_ylim(1.5, 3.7)
    save(fig, "fig2_envelope")


# ============================================================ FIG 3 ===========
# Four-stage funnel per duration (1/4/24 h): facility total power -> eligible load
# (flexible clusters only, the SCOPE step) -> after requiring persistence -> after
# accounting for coincidence. The scope step is ~-93%, so a linear funnel would
# crush the lower stages; we therefore use a BROKEN axis (method B): the facility
# bar sits above a zig-zag break at a capped width, and the lower three stages are
# drawn at a common true-MW scale so the persistence and coincidence steps stay
# visible. Colours: matplotlib "Set2" (one hue per stage). No on-figure title, no
# panel tags, no bottom legend, no Shapley range; every reduction is phrased as
# "after <effect>, -X%" so it cannot be misread as "X% of <effect>".
S2 = plt.get_cmap("Set2").colors               # 8 pastel, colour-blind-safe hues
F3_FAC, F3_ELI, F3_PERS, F3_COINC = S2[0], S2[1], S2[2], S2[3]

def _blend(c0, c1):
    return tuple((a + b) / 2 for a, b in zip(c0, c1))

def fig3():
    # Pure-shape version: no on-figure text, no whiskers, no labels. Three equal
    # panels (1/4/24 h). Facility bar is widened until neighbouring panels' bars
    # nearly touch; the lower three stages are ~1.7x wider than before. The zig-zag
    # break is clipped to the trapezoid's own width so it never juts past the shape.
    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    WSCALE = 0.85 / D["f3"][1]["P1"]         # ~1.7x wider lower stages
    FAC = D["facility_mean"]
    FAC_W = 1.52                             # facility half-width (bars nearly touch)
    XLIM = 1.58
    yF, yA, yP, yS = 4.0, 3.0, 2.0, 1.0     # stage centre heights
    hh = 0.30                                # half stage-band height
    ybrk = (yF + yA) / 2                     # break marker height

    def zigzag(xhalf, y, n=6, amp=0.055):
        xs = np.linspace(-xhalf, xhalf, 2 * n + 1)
        ys = y + amp * np.array([1 if k % 2 else -1 for k in range(len(xs))])
        return xs, ys

    for ax, T in zip(axes, (1, 4, 24)):
        d = D["f3"][T]
        low_vals = [d["P1"], d["P2"], d["P3"]]
        low_ys = [yA, yP, yS]
        low_cols = [F3_ELI, F3_PERS, F3_COINC]
        # lower true-scale sub-funnel: eligible -> persistence -> sustained
        for (y0, v0), (y1, v1), c in (((yA, low_vals[0]), (yP, low_vals[1]),
                                       _blend(F3_ELI, F3_PERS)),
                                      ((yP, low_vals[1]), (yS, low_vals[2]),
                                       _blend(F3_PERS, F3_COINC))):
            w0, w1 = v0 * WSCALE, v1 * WSCALE
            ax.add_patch(Polygon([(-w0, y0 - hh), (w0, y0 - hh),
                                  (w1, y1 + hh), (-w1, y1 + hh)],
                                 closed=True, fc=c, ec="none", alpha=0.70, zorder=3))
        # facility funnel wall down to the eligible bar (broken axis)
        w_eli = d["P1"] * WSCALE
        ax.add_patch(Polygon([(-FAC_W, yF - hh), (FAC_W, yF - hh),
                              (w_eli, yA + hh), (-w_eli, yA + hh)],
                             closed=True, fc=_blend(F3_FAC, F3_ELI), ec="none",
                             alpha=0.45, zorder=2))
        # zig-zag break, clipped to the trapezoid half-width at the break height
        w_brk = FAC_W + (w_eli - FAC_W) * ((yF - hh - ybrk) / (yF - hh - (yA + hh)))
        for off in (-0.045, 0.045):
            xs, ys = zigzag(w_brk * 0.9, ybrk + off)
            ax.fill_between(xs, ys - 0.02, ys + 0.02, color="white", lw=0, zorder=5)
            ax.plot(xs, ys, color="#8a8a8a", lw=0.9, zorder=6, solid_capstyle="round")
        # stage bars (no text)
        for y, v, c, w in ([(yF, FAC, F3_FAC, FAC_W)] +
                           list(zip(low_ys, low_vals, low_cols,
                                    [v * WSCALE for v in low_vals]))):
            ax.add_patch(plt.Rectangle((-w, y - hh), 2 * w, 2 * hh, fc=c,
                         ec="white", lw=0.9, zorder=7))
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(0.5, 4.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=0.2)
    save(fig, "fig3_decomposition")


# ============================================================ FIG 4 ===========
def fig4():
    fig = plt.figure(figsize=(7.4, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.05], wspace=0.5)
    axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1]); axc = fig.add_subplot(gs[2])
    comp = D["f4_comp"]; good = D["f4_good"]
    ix = np.where(good)[0]
    x = np.arange(len(ix))
    # (a) three-category stack for the main paper (7-layer detail -> SI)
    lp = comp["lp_active"][ix]
    hp = comp["hp_shift_active"][ix]
    other = (comp["other_nongpu"] + comp["facility_overhead"] + comp.get("host", 0.0)
             + comp["gpu_idle"] + comp["floor_active"])[ix]
    stack = np.vstack([other, hp, lp])
    labels = ["Other / non-curtailable facility demand", "High-priority training",
              "Low-priority workload eligible for curtailment"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # tab10: blue / orange / green
    axa.stackplot(x, stack, colors=cols, labels=labels, lw=0, zorder=2)
    axa.set_xlim(0, len(ix)); axa.set_ylim(0, np.nansum(stack, 0).max() * 1.02)
    axa.set_xlabel("Hour in 185-day record")
    axa.set_ylabel("Power (MW)")
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.tick_params(length=3)
    # (b) distribution (histogram) of the hourly low-priority share
    samp = D["f4_share_ecdf_x"] * 100          # 200 equal-probability quantiles
    edges = np.linspace(0, 15, 26)
    h, _ = np.histogram(samp, bins=edges)
    h = h / h.sum() * 100.0                     # percent of observed hours per bin
    centers = 0.5 * (edges[:-1] + edges[1:])
    axb.bar(centers, h, width=(edges[1] - edges[0]) * 0.92, color="#d62728",
            edgecolor="white", lw=0.3, zorder=3)
    axb.axvline(D["f4_share_mean"] * 100, color="#9467bd", ls=FIX_DASH, lw=LW_REF,
                zorder=4, label=f"Mean = {D['f4_share_mean']*100:.0f}%")
    axb.set_xlim(0, 15); axb.set_ylim(0, h.max() * 1.15)
    axb.set_xlabel("Low-priority share (%)")
    axb.set_ylabel("Frequency (%)")
    axb.yaxis.set_major_locator(MultipleLocator(5))
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.grid(axis="y", color=GRIDCOL, lw=0.6, zorder=0); axb.tick_params(length=3)
    axb.legend(loc="upper right", frameon=False, handlelength=1.4, fontsize=6.4)
    # (c) scope expansion via HP-training participation
    FR = D["f7_frac"]; y95 = D["f7_oper"][0.95]
    steps = [0, 0.25, 0.5, 1.0]
    vals = [float(np.interp(f, FR, y95)) for f in steps]
    base = vals[0]
    xlab = ["none", "+25%", "+50%", "+100%"]
    xc = np.arange(len(steps))
    axc.bar(xc, [base] * len(steps), width=0.66, color="#8c564b", edgecolor="white",
            lw=0.6, zorder=3, label="Existing low-priority scope")
    axc.bar(xc, [v - base for v in vals], bottom=[base] * len(steps), width=0.66,
            color="#e377c2", edgecolor="white", lw=0.6, zorder=3,
            label="Added high-priority training")
    for xi, v in zip(xc, vals):
        axc.text(xi, v + max(vals) * 0.02, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=6.6, color="#333")
    axc.set_xticks(xc); axc.set_xticklabels(xlab, fontsize=7.0)
    axc.set_ylabel("Power (MW)")
    axc.set_ylim(0, max(vals) * 1.32)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.grid(axis="y", color=GRIDCOL, lw=0.6, zorder=0); axc.tick_params(length=3)
    axc.legend(loc="upper left", frameon=False, handlelength=1.1, fontsize=6.1,
               labelspacing=0.3, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig4_scope")


# ============================================================ FIG 5 ===========
def fig5():
    # Two panels. (a) time-domain EVIDENCE that cluster low-load periods pile up on
    # the same days; (b) the coincidence factor vs event duration against an
    # unrelated-timing null, with the second operator overlaid. The MW/percent cost
    # of this coincidence is deliberately NOT re-drawn here -- that is Fig. 2 (the
    # measured vs independent envelope) and Fig. 6a (the coincidence overstatement).
    fig = plt.figure(figsize=(8.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.34)
    axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

    # (a) distribution of how many clusters are low on the same day -------------
    # Observed vs an independent-timing model. The independent model keeps each
    # cluster's own marginal chance of a low-load day but breaks the cross-cluster
    # timing (Poisson-binomial of the per-cluster marginals, evaluated by seeded
    # Monte Carlo). Derived purely from the cached f5_short_clusters array.
    M = D["f5_short_clusters"]                        # 13 x ndays, NaN where inactive
    low = M > 0                                       # cluster low on that day
    cnt = np.where(np.any(~np.isnan(M), axis=0),      # clusters low per day
                   np.nansum(low.astype(float), axis=0), np.nan)
    valid = ~np.isnan(cnt)
    cnt = cnt[valid].astype(int)
    kmax = int(cnt.max())
    kk = np.arange(kmax + 1)
    obs = np.array([(cnt == k).mean() for k in kk])
    p_i = np.array([np.nanmean(low[i]) for i in range(low.shape[0])])   # marginals
    rng = np.random.default_rng(2027)
    sim = (rng.random((40000, low.shape[0])) < p_i).sum(axis=1)
    ind = np.array([(sim == k).mean() for k in kk])
    w = 0.42
    # Distinct colour per quantity (no shared blue/grey across the figure).
    A_OBS   = "#E69F00"      # observed distribution (amber, emphasised)
    A_OBS_E = "#B5730A"      # observed envelope (deeper amber, close to A_OBS)
    A_IND   = "#DCD3E6"      # unrelated-timing distribution (light lavender)
    A_IND_E = "#C7B8DC"      # unrelated envelope (light, de-emphasised)
    axa.bar(kk - w / 2, obs, w, color=A_OBS, label="Observed")
    axa.bar(kk + w / 2, ind, w, color=A_IND,
            label="If low-load days were unrelated")
    ytop = max(obs.max(), ind.max())
    # envelope outlines tracing each distribution's shape, so the reader can
    # compare the fatter-both-ends observed shape against the concentrated one.
    axa.plot(kk, obs, color=A_OBS_E, lw=1.7, marker="o", ms=3.0, zorder=5)
    axa.plot(kk, ind, color=A_IND_E, lw=1.3, ls=INDEP_DASH, marker="s", ms=2.6,
             zorder=5)
    axa.set_xticks(kk)
    axa.set_ylim(0, ytop * 1.18)
    axa.set_xlabel("Number of clusters in a low-load state on the same day")
    axa.set_ylabel("Frequency of days (%)")
    axa.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.grid(axis="y", color=GRIDCOL, lw=0.6, zorder=0)
    axa.tick_params(length=3)
    axa.legend(loc="upper right", frameon=True, handlelength=1.2, fontsize=6.6,
               facecolor="white", edgecolor="black", framealpha=1.0).get_frame().set_linewidth(0.8)

    # (b) coincidence factor vs duration, two operators, unrelated-timing null ----
    B_NULL = "#CFE3C4"       # unrelated-timing range (light green)
    B_BOOT = "#F3C6D3"       # 90% bootstrap interval (light pink)
    B_WORK = "#0072B2"       # this work, 13 clusters (blue)
    B_2ND  = "#009E73"       # second operator, 4 sites (green)
    nb = D["f5_null"]
    axb.fill_between(Gx, nb["p5"][sel], nb["p95"][sel], color=B_NULL, alpha=0.9,
                     lw=0)
    cbn = D["f5_coinc_boot"]
    axb.fill_between(Gx, cbn["p5"][sel], cbn["p95"][sel], color=B_BOOT,
                     alpha=0.8, lw=0)
    axb.plot(Gx, D["f5_coinc_meas"][sel], color=B_WORK, lw=LW_MAIN)
    hel = D["f5_helios"]; Th = [1, 4, 24]
    axb.plot(Th, [hel[t] for t in Th], color=B_2ND, lw=LW_REF, marker="D", ms=5)
    axb.axhline(1.0, color="#9AA3AC", lw=0.9, ls=(0, (1, 2)))
    dur_axis(axb, "Coincidence factor")
    axb.set_ylim(0.85, 1.03)
    save(fig, "fig5_coincidence")


# ============================================================ FIG 6 ===========
def fig6():
    # Relative-overstatement view: each curve is the percentage by which a more
    # optimistic power estimate exceeds the next, more conservative one.
    #   persistence curve = (P1/P2 - 1) : average load over persistence-adjusted
    #   coincidence curve = (P2/P3 - 1) : persistence-adjusted over sustained
    # Teal = the coincidence step measured on an independent second operator.
    # No on-figure text except the two axes; title/legend live in the caption.
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bp = D["f6_boot_over"][0.95]["persist"]; bc = D["f6_boot_over"][0.95]["coinc"]
    ax.fill_between(Gx, bp["p5"][sel], bp["p95"][sel], color=C_PERS, alpha=0.16, lw=0)
    ax.fill_between(Gx, bc["p5"][sel], bc["p95"][sel], color=C_COINC, alpha=0.16, lw=0)
    ax.plot(Gx, D["f6_persOver"][0.95][sel], color=C_PERS, lw=LW_MAIN)
    ax.plot(Gx, D["f6_coincOver"][0.95][sel], color=C_COINC, lw=LW_MAIN)
    # external check: connect the three horizons so they read as a curve
    Th = [1, 4, 24]
    ax.plot(Th, [D["f6_helios_coincOver"][t] for t in Th], color=C_HEL,
            lw=1.6, marker="D", ms=4.5, mfc="none", mew=1.2)
    dur_axis(ax, "Relative overstatement (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylim(0, 60); ax.set_xlim(0.92, 25.5)
    ax.xaxis.label.set_size(11); ax.yaxis.label.set_size(11)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "fig6_driver")


# ============================================================ FIG 7 ===========
# No A/B/C stakeholder headers or panel tags (author request 2026-09-17).
# tab10 palette: every distinct quantity across all three panels gets its own
# colour, and no colour is reused between panels (blue appears once only).
TAB10 = plt.get_cmap("tab10").colors
F7C = {
    "envelope": TAB10[0],   # measured flexibility envelope   (blue)
    "fixed":    TAB10[7],   # fixed-share assumption          (gray, dashed)
    "noflex":   TAB10[3],   # no flexibility                  (red, dash-dot)
    "full":     TAB10[6],   # fully curtailable reference     (pink)
    "T1":       TAB10[1],   # 1-hour event                    (orange)
    "T4":       TAB10[2],   # 4-hour event                    (green)
    "T24":      TAB10[4],   # 24-hour event                   (purple)
    "c90":      TAB10[5],   # 90% coverage                    (brown)
    "c95":      TAB10[8],   # 95% coverage                    (olive)
    "c99":      TAB10[9],   # 99% coverage                    (cyan)
}
F7_FRAME = dict(frameon=True, framealpha=0.95, edgecolor="#B0B0B0",
                fancybox=False, handlelength=1.7)


def fig7():
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.6, 2.7))
    facility = D["facility_mean"]
    # (a) planner: additional connectable load (MW) from the A8 peak-cap
    # interconnection screen (see prep_figs.py). The measured / fixed-share /
    # no-flex lines are ~flat in the budget because the resource is bound by its
    # worst hour, not by the annual curtailment-hour budget.
    bud = D["f7a_budget_h"]; o = np.argsort(bud)
    axa.plot(bud[o], D["f7a_full"][o], color=F7C["full"], lw=LW_MAIN,
             marker="o", ms=3.5, label="Fully curtailable")
    axa.plot(bud[o], D["f7a_fixed"][o], color=F7C["fixed"], ls=FIX_DASH,
             lw=LW_REF, marker="o", ms=3.0, label="Fixed-share assumption")
    axa.fill_between(bud[o], D["f7a_measured_p5"][o], D["f7a_measured_p95"][o],
                     color=F7C["envelope"], alpha=0.18, lw=0)
    axa.plot(bud[o], D["f7a_measured"][o], color=F7C["envelope"], lw=LW_MAIN,
             marker="o", ms=3.5, label="Measured flexibility envelope")
    axa.plot(bud[o], D["f7a_noflex"][o], color=F7C["noflex"], ls=INDEP_DASH,
             lw=LW_REF, marker="o", ms=3.0, label="No flexibility")
    axa.set_xscale("log")
    # data spans ~22-175 h/year (budgets 0.25-2.0% of 8760); tick within range
    axa.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200]))
    axa.xaxis.set_minor_locator(NullLocator())
    axa.set_xticklabels(["20", "50", "100", "200"])
    axa.set_xlim(18, 210)
    axa.set_xlabel("Annual curtailment allowance (hours/year)")
    axa.set_ylabel("Additional connectable load (MW)")
    _ymax = float(np.max(D["f7a_full"])) * 1.08
    axa.set_ylim(0, _ymax)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.grid(color=GRIDCOL, lw=0.6, zorder=0); axa.tick_params(length=3)
    # (b) operations
    cov = D["f7_cov"] * 100
    for T, key in ((1, "T1"), (4, "T4"), (24, "T24")):
        axb.plot(cov, D["f7_grid"][T], color=F7C[key], lw=LW_MAIN,
                 label=f"{T}-hour event")
    axb.set_xlabel("Required coverage (%)")
    axb.set_ylabel("Sustained curtailable power (MW)")
    axb.set_xlim(80, 99.9)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.grid(color=GRIDCOL, lw=0.6, zorder=0); axb.tick_params(length=3)
    # (c) recruitment
    FR = D["f7_frac"] * 100
    for a, key in ((0.90, "c90"), (0.95, "c95"), (0.99, "c99")):
        axc.plot(FR, D["f7_oper"][a], color=F7C[key], lw=LW_MAIN,
                 label=f"{int(a*100)}% coverage")
    axc.set_xlabel("High-priority training enrolled (%)")
    axc.set_ylabel("4-hour sustained curtailable power (MW)")
    axc.set_xlim(0, 100)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.grid(color=GRIDCOL, lw=0.6, zorder=0); axc.tick_params(length=3)
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig7_decisions")


fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
print("ALL DONE ->", OUT)
