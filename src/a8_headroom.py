#!/usr/bin/env python
"""A8 (v2, revision 2): illustrative peak-cap screening on MISO South.

RULE (exactly what the code enforces, for the Methods text):
  Let D_t be regional demand (EIA-930 sub-region 8910, hourly, 2024-2025;
  gaps <= 48 h interpolated), P_cap = max_t D_t * (1 + m).  A candidate load L
  (MW, = "nameplate" = the datacenter's 7-day rolling-max facility draw) has
  hourly draw L*s_t and hourly curtailable capability L*c_t.  Residual excess
  r_t = max(0, D_t + L*s_t - P_cap).  L is ADMISSIBLE iff
      (i)  r_t <= L*c_t for EVERY hour t  (no post-response violation), and
      (ii) #{t : r_t > 0} <= budget (hours per year in which response is invoked).
  Headroom = sup admissible L (bisection).  Scenarios differ only in c_t:
      full      c_t = s_t            (Duke "Rethinking Load Growth": fully curtailable)
      const20   c_t = 0.20*s_t       (DCFlex / Dvorkin-style interruptible tier)
      constM    c_t = m_bar*s_t      (constant share = our mean eligible share)
      measured  c_t = our hourly eligible curtailment (idle_retained boundary)
      measured_attributed  same with the attributed boundary (upper bound)
  Time alignment: the 185-day trace (4,440 h) is circularly rolled by a random
  offset and tiled (np.resize) to the 17,544 system hours; 200 offsets ->
  P5/P50/P95.  Missing trace hours are linearly interpolated before tiling.
  Facility conversion of c_t: marginal (x1.0) and average PUE (x cfg.pue).
This is a SCREENING ILLUSTRATION (no accredited supply, outages, imports,
transfer limits, N-1); see README caveats.
"""
import json
import os
import sys

import numpy as np
import matplotlib

PLOT_ONLY = os.environ.get("A8_PLOT_ONLY") == "1"
if not PLOT_ONLY:
    import pandas as pd
    import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
A1 = os.environ.get("A1_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a1_out"))
A5 = os.environ.get("A5_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a5_out"))
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a8_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(os.path.dirname(__file__), "power_curves.yaml"))
SYS = os.environ.get("SYS_CSV", os.path.join(AGG, "miso_south_hourly_2024_2025.csv"))
os.makedirs(OUT, exist_ok=True)
N_ALIGN = int(os.environ.get("A8_NALIGN", 200))
BUDGETS_PCT = [0.25, 0.5, 1.0, 2.0]
MARGINS = [0.0, 0.02, 0.05, 0.10]
SCEN = ["full", "const20", "constM", "measured", "measured_attributed"]
HYPERION_MW = 2000.0
EVENT_MARGIN = 0.05
EVENT_BUDGET_PCT = 0.5


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def plot_headroom_by_scenario(summary):
    """Render the main A8 panel from stored quantile bands.

    Keeping this plot independent of the expensive offset sweep allows label-only
    corrections to be reproduced without rerunning the screening calculation.
    """
    cols = {"full": "#999", "const20": "tab:orange", "constM": "tab:blue", "measured": "tab:green",
            "measured_attributed": "tab:olive"}
    labels = {"full": "fully curtailable (Duke assumption)", "const20": "20% interruptible tier (DCFlex-style)",
              "constM": "constant share = our mean eligible share", "measured": "hourly eligibility, idle-retained (this work)",
              "measured_attributed": "hourly eligibility, attributed boundary (upper bound)"}
    budget_pct = EVENT_BUDGET_PCT
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = [m * 100 for m in MARGINS]
    for tname, ls in [("facility_marginal_pue1.0", "-"), ("facility_avg_pue", "--")]:
        tier = summary["tiers"].get(tname)
        if tier is None:
            continue
        if ls == "-":
            ax.plot(xs, [tier["headroom_no_flex_mw"][f"m{m}"]["p50"] for m in MARGINS],
                    "k:", lw=1.2, label="no flexibility (noncurtailable load)")
        for sc in SCEN:
            bands = [tier["headroom_mw"][f"m{m}"][sc][f"{budget_pct}%"] for m in MARGINS]
            m50 = [v["p50"] for v in bands]
            lo = [v["p50"] - v["p5"] for v in bands]
            hi = [v["p95"] - v["p50"] for v in bands]
            ax.errorbar(xs, m50, yerr=[lo, hi], fmt="o" + ls, color=cols[sc], capsize=2, ms=4,
                        label=labels[sc] + ("" if ls == "-" else " (avg-PUE tier)")
                        if (ls == "-" or sc == "measured") else None)
    ax.axhline(HYPERION_MW, color="k", ls="--", lw=1, label="2 GW analytical reference")
    ax.set(xlabel="capacity margin above historical peak (%)", ylabel="admissible AI-DC load (MW)",
           title=f"MISO South peak-cap screening, {budget_pct}% invoked-hours budget")
    ax.legend(fontsize=7)
    savefig(fig, "f1_headroom_by_scenario.png")


def plot_headroom_ratio(summary):
    """Render the A8 ratio heatmaps from stored medians at two-column font size."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, key, ttl in [
            (axes[0], "measured_over_constM", "hourly / constant mean"),
            (axes[1], "measured_over_const20", "hourly / 20% tier")]:
        matrix = np.array([
            [summary["tiers"]["facility_marginal_pue1.0"]["ratios_p50_unrounded"]
             [f"m{m}"][f"{b}%"][key] for b in BUDGETS_PCT]
            for m in MARGINS])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(BUDGETS_PCT)), [f"{b}%" for b in BUDGETS_PCT], fontsize=22)
        ax.set_yticks(range(len(MARGINS)), [f"{m*100:.0f}%" for m in MARGINS], fontsize=22)
        ax.set_ylabel("capacity margin" if ax is axes[0] else "", fontsize=24)
        ax.set_title(ttl, fontsize=22)
        for i in range(len(MARGINS)):
            for j in range(len(BUDGETS_PCT)):
                color = "white" if matrix[i, j] < 0.7 or matrix[i, j] > 1.3 else "black"
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=22, color=color)
    fig.supxlabel("invoked-hours budget", fontsize=24)
    cbar = fig.colorbar(im, ax=axes)
    cbar.set_label("headroom ratio", fontsize=24)
    cbar.ax.tick_params(labelsize=22)
    savefig(fig, "f2_headroom_ratio.png")


def load_system():
    d = pd.read_csv(SYS)
    d["demand_mw"] = pd.to_numeric(d["demand_mw"], errors="coerce")
    d["utc"] = pd.to_datetime(d["utc_end"], format="%m/%d/%Y %I:%M:%S %p")
    d = d.sort_values("utc").drop_duplicates("utc")
    full = pd.date_range(d["utc"].min(), d["utc"].max(), freq="h")
    s = d.set_index("utc")["demand_mw"].reindex(full).interpolate(limit=48).bfill().ffill()
    return s


def load_dc(cfg, pue_factor):
    fl = pd.read_parquet(os.path.join(A1, "fleet_hourly_power.parquet"))
    env = pd.read_parquet(os.path.join(A5, "envelope_hourly.parquet"))
    F = fl["base_mw"].to_numpy()
    good = fl["have"].to_numpy().astype(bool) & env["good"].to_numpy().astype(bool)
    t = np.arange(len(F))
    F = np.interp(t, t[good], F[good])
    C = {}
    for b in ["idle_retained", "attributed"]:
        c = env[f"curtail_{b}"].to_numpy()
        C[b] = np.interp(t, t[good], c[good]) * pue_factor
    cap = pd.Series(F).rolling(168, min_periods=24).max().bfill().to_numpy()
    return F / cap, {b: c / cap for b, c in C.items()}, int(good.sum())


def feasible(D, Pcap, s_al, avail, budget_h, L):
    r = np.maximum(0.0, D + L * s_al - Pcap)
    return bool(np.all(r <= L * avail + 1e-9) and int((r > 0).sum()) <= budget_h), r


def max_L(D, Pcap, s_al, avail, budget_h, hi=20000.0):
    if not feasible(D, Pcap, s_al, avail, budget_h, 1.0)[0]:
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(D, Pcap, s_al, avail, budget_h, mid)[0]:
            lo = mid
        else:
            hi = mid
    return lo


def quantile_band(v):
    v = np.asarray(v, dtype=float)
    return {"p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95))}


def event_structure_across_alignments(D, Pcap, s, c, offsets, L_by_alignment,
                                      budget_h):
    """Summarise response events using each alignment's own admissible L.

    This avoids the previous mismatch in which the median L across alignments
    was evaluated on offsets[0].  That single arbitrary alignment was neither
    the median alignment nor a distributional result.
    """
    n_events, hours, max_event_h, median_event_h = [], [], [], []
    curtailed_energy_pct, max_depth_frac = [], []
    infeasible = []
    for i, (off, L) in enumerate(zip(offsets, L_by_alignment)):
        s_al = np.resize(np.roll(s, int(off)), len(D))
        c_al = np.resize(np.roll(c, int(off)), len(D))
        ok, r = feasible(D, Pcap, s_al, c_al, budget_h, float(L))
        if not ok:
            infeasible.append(i)
            continue
        on = r > 0
        edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
        durs = np.where(edges == -1)[0] - np.where(edges == 1)[0]
        n_events.append(len(durs))
        hours.append(int(on.sum()))
        max_event_h.append(int(durs.max()) if len(durs) else 0)
        median_event_h.append(float(np.median(durs)) if len(durs) else 0.0)
        curtailed_energy_pct.append(float(r.sum() / max((L * s_al).sum(), 1e-9) * 100))
        max_depth_frac.append(float((r / np.maximum(L * s_al, 1e-9)).max()))
    if infeasible:
        raise RuntimeError(f"{len(infeasible)} event-structure alignments were infeasible: {infeasible[:10]}")
    return {
        "definition": ("Each random trace-to-system offset is evaluated at its own admissible measured-scenario "
                       "headroom L for margin=5% and invoked-hours budget=0.5%; P5/P50/P95 are across offsets."),
        "n_alignments": int(len(offsets)),
        "all_alignments_feasible": True,
        "budget_hours": float(budget_h),
        "L_mw": quantile_band(L_by_alignment),
        "n_events": quantile_band(n_events),
        "hours_invoked": quantile_band(hours),
        "max_event_h": quantile_band(max_event_h),
        "median_event_h": quantile_band(median_event_h),
        "curtailed_energy_pct_of_dc": quantile_band(curtailed_energy_pct),
        "max_depth_frac_of_load": quantile_band(max_depth_frac),
    }


def tier_inputs(cfg, pf):
    if isinstance(pf, tuple):
        s, c, ngood = load_dc(cfg, 1.0)
        c = {b: v * (1 + pf[1] * s) for b, v in c.items()}
        pf_label = f"1+{pf[1]}*s_t"
    else:
        s, c, ngood = load_dc(cfg, pf)
        pf_label = pf
    return s, c, ngood, pf_label


def update_event_structure_only(cfg, D, tiers):
    """Lightweight refresh of the alignment-level event statistics."""
    summary_path = os.path.join(OUT, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"A8_EVENT_ONLY requires an existing {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    nH = len(D)
    Pcap = float(D.max()) * (1 + EVENT_MARGIN)
    budget_h = EVENT_BUDGET_PCT / 100.0 * nH
    for tname, pf in tiers.items():
        if tname not in summary["tiers"]:
            log(f"event-only {tname}: skipped (tier absent from existing summary)")
            continue
        s, c, _, _ = tier_inputs(cfg, pf)
        offsets = np.random.default_rng(8).integers(0, len(s), N_ALIGN)
        Ls = []
        for off in offsets:
            s_al = np.resize(np.roll(s, int(off)), nH)
            c_al = np.resize(np.roll(c["idle_retained"], int(off)), nH)
            Ls.append(max_L(D, Pcap, s_al, c_al, budget_h))
        summary["tiers"][tname]["event_structure_measured_m0.05_0.5pct"] = \
            event_structure_across_alignments(D, Pcap, s, c["idle_retained"],
                                              offsets, np.asarray(Ls), budget_h)
        log(f"event-only {tname}: {N_ALIGN} alignments")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A8 v2 MISO South screening — summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    log("DONE A8 event-only refresh")


def run(D, s, c, rng):
    nH, nT = len(D), len(s)
    Pcap0 = float(D.max())
    tile = lambda x, off: np.resize(np.roll(x, off), nH)
    mean_share = float(c["idle_retained"].mean() / s.mean())
    offsets = rng.integers(0, nT, N_ALIGN)
    res = {m: {sc: {b: np.zeros(N_ALIGN) for b in BUDGETS_PCT} for sc in SCEN} for m in MARGINS}
    noflex = {m: np.zeros(N_ALIGN) for m in MARGINS}
    for i, off in enumerate(offsets):
        s_al = tile(s, off)
        avail = {"full": s_al, "const20": 0.20 * s_al, "constM": mean_share * s_al,
                 "measured": tile(c["idle_retained"], off), "measured_attributed": tile(c["attributed"], off)}
        for m in MARGINS:
            cap = Pcap0 * (1 + m)
            noflex[m][i] = float(np.min((cap - D) / np.maximum(s_al, 1e-6)))
            for b in BUDGETS_PCT:
                bh = b / 100.0 * nH
                for sc in SCEN:
                    res[m][sc][b][i] = max_L(D, cap, s_al, avail[sc], bh)
        if (i + 1) % 50 == 0:
            log(f"alignment {i+1}/{N_ALIGN}")
    return res, noflex, mean_share, offsets, Pcap0


def main():
    if PLOT_ONLY:
        summary_path = os.path.join(OUT, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"A8_PLOT_ONLY requires an existing {summary_path}")
        with open(summary_path) as f:
            summary = json.load(f)
        plot_headroom_by_scenario(summary)
        plot_headroom_ratio(summary)
        log("DONE A8 plot-only refresh")
        return
    cfg = yaml.safe_load(open(CFG))
    rng = np.random.default_rng(8)
    D = load_system().to_numpy()
    nH = len(D)
    band = quantile_band
    tiers = {"facility_marginal_pue1.0": 1.0, "facility_avg_pue": cfg["pue"]["base"],
             # item 10: load-following marginal cooling: c_t x (1 + k*s_t), k = 0.1 (cooling ~10% of IT at full load)
             "facility_marginal_linear_k0.1": ("linear", 0.10)}
    if os.environ.get("A8_EVENT_ONLY") == "1":
        update_event_structure_only(cfg, D, tiers)
        return
    summary = {"rule": __doc__.split("RULE")[1].split("This is a SCREENING")[0].strip(),
               "system": {"region": "MISO South (EIA-930 sub-region 8910: AR/LA/MS/E-TX)", "hours": nH,
                          "years": nH / 8766.0, "peak_mw": float(D.max()), "mean_mw": float(D.mean()),
                          "min_mw": float(D.min())},
               "settings": {"n_align": N_ALIGN, "budgets_pct_hours": BUDGETS_PCT, "cap_margins": MARGINS,
                            "cap_rule": "historical 2-year peak x (1 + margin)",
                            "dc_norm": "7-day rolling max of A1 base facility MW",
                            "tiling": "circular roll by random offset, np.resize to system length",
                            "hyperion_mw": HYPERION_MW},
               "tiers": {}}
    results = {}
    for tname, pf in tiers.items():
        s, c, ngood, pf_label = tier_inputs(cfg, pf)
        res, noflex, mean_share, offsets, Pcap0 = run(D, s, c, np.random.default_rng(8))
        results[tname] = (res, noflex, s, c, offsets)
        med = lambda m, sc, b: float(np.median(res[m][sc][b]))
        summary["tiers"][tname] = {
            "pue_factor": pf_label, "trace_hours_used": ngood,
            "dc_profile": {"mean_draw_frac_of_nameplate": float(s.mean()),
                           "eligible_curtail_mean_frac_of_nameplate": float(c["idle_retained"].mean()),
                           "eligible_share_of_load": mean_share,
                           "attributed_share_of_load": float(c["attributed"].mean() / s.mean())},
            "headroom_mw": {f"m{m}": {sc: {f"{b}%": band(res[m][sc][b]) for b in BUDGETS_PCT} for sc in SCEN} for m in MARGINS},
            "headroom_no_flex_mw": {f"m{m}": band(noflex[m]) for m in MARGINS},
            "ratios_p50_unrounded": {f"m{m}": {f"{b}%": {
                "measured_over_noflex": med(m, "measured", b) / max(float(np.median(noflex[m])), 1e-9),
                "measured_over_constM": med(m, "measured", b) / max(med(m, "constM", b), 1e-9),
                "measured_over_const20": med(m, "measured", b) / max(med(m, "const20", b), 1e-9),
                "measured_over_full": med(m, "measured", b) / max(med(m, "full", b), 1e-9),
                "attributed_over_measured": med(m, "measured_attributed", b) / max(med(m, "measured", b), 1e-9),
                "const20_over_measured": med(m, "const20", b) / max(med(m, "measured", b), 1e-9),
                "full_over_measured": med(m, "full", b) / max(med(m, "measured", b), 1e-9),
                "measured_in_hyperion_units": med(m, "measured", b) / HYPERION_MW}
                for b in BUDGETS_PCT} for m in MARGINS}}
        # Event structure across all random alignments.  Each offset uses its
        # corresponding admissible L; no arbitrary offset/median-L mismatch.
        M0, B0 = EVENT_MARGIN, EVENT_BUDGET_PCT
        summary["tiers"][tname]["event_structure_measured_m0.05_0.5pct"] = \
            event_structure_across_alignments(
                D, Pcap0 * (1 + M0), s, c["idle_retained"], offsets,
                res[M0]["measured"][B0], B0 / 100.0 * nH)
        pd.DataFrame([{"tier": tname, "margin": m, "scenario": sc, "budget_pct": b, **band(res[m][sc][b])}
                      for m in MARGINS for sc in SCEN for b in BUDGETS_PCT]).to_csv(
            os.path.join(OUT, f"headroom_table_{tname}.csv"), index=False)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A8 v2 MISO South screening — summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    log(json.dumps({t: v["ratios_p50_unrounded"]["m0.05"]["0.5%"] for t, v in summary["tiers"].items()}, indent=1))

    # ---- figures (marginal tier main; avg-PUE dashed) ----------------------------
    plot_headroom_by_scenario(summary)
    B0 = EVENT_BUDGET_PCT

    res = results["facility_marginal_pue1.0"][0]
    plot_headroom_ratio(summary)

    s, c, offsets = results["facility_marginal_pue1.0"][2], results["facility_marginal_pue1.0"][3], results["facility_marginal_pue1.0"][4]
    Lvals = res[EVENT_MARGIN]["measured"][EVENT_BUDGET_PCT]
    L0 = float(np.median(Lvals))
    i_rep = int(np.argmin(np.abs(Lvals - L0)))
    s_al = np.resize(np.roll(s, offsets[i_rep]), nH)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ldc = np.sort(D)[::-1]
    pct = np.linspace(0, 100, nH)
    ax.plot(pct, ldc, "k", lw=1, label="MISO South demand 2024–25")
    ax.axhline(D.max() * 1.05, color="r", ls="--", lw=1, label="cap = peak + 5%")
    tot = np.sort(D + L0 * s_al)[::-1]
    ax.plot(pct, tot, color="tab:green", lw=1, label=f"+ {L0:.0f} MW AI-DC (measured, 0.5% budget)")
    ax.set_xlim(0, 3); ax.set_ylim(ldc[int(0.03 * nH)] * 0.98, tot[0] * 1.02)
    ax.set(xlabel="% of hours", ylabel="MW", title="top 3% of hours")
    ax.legend(fontsize=8)
    savefig(fig, "f3_load_duration_top.png")
    log("DONE a8_headroom v2")


if __name__ == "__main__":
    main()
