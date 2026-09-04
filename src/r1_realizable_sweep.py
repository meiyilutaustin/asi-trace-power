#!/usr/bin/env python
"""R1 (reviewer-requested robustness check): realizable-fraction and interruption-cost
sweep for the eligible-curtailment layer.

The v2 headline (3.55 MW idle-retained, K(0.95,4h) = 2.32 MW) assumes every
label-defined LP GPU-hour can be preempted immediately and for free.  This
script sweeps the three parameters that assumption hides, and reports what
survives.

Parameters
  q      realizable fraction of the eligible layer that a scheduler would in
         fact release on call (SLA-safe, checkpointable, not gang-blocked).
         q = 1 reproduces the v2 headline.
  rho    fraction of the interrupted work that must be RE-EXECUTED afterwards
         (rho = 0: best-effort work is dropped, as spot semantics allow;
          rho = 1: every interrupted GPU-hour is redone).
  T      checkpoint interval (h).  Work since the last checkpoint is lost, on
         average T/2 per preempted pod, and must be redone on top of the
         deferred work.  OSDI'26 reports a 60 s graceful-eviction budget
         (mean 13 s, P95 48 s), so the ramp-down itself is sub-hourly and is
         not modelled here.

Reported
  1. Delivered power during an event: K_q(a,h) = q * K(a,h) exactly (the
     running-min quantile is positively homogeneous); verified numerically.
  2. Net energy multiplier: over an h-hour call plus its recovery,
        E_net / E_gross = 1 - rho * (1 + T / (2h)),
     i.e. how much of the energy relief survives the rebound.  Negative means
     the call raises total energy (the "flexible DCs can raise emissions"
     regime).
  3. Recovery time R_min: hours needed to absorb the rebound from the SOLD
     product K_q(0.95,h) = q*K(0.95,h), at a rebound rate capped at a fixed
     fraction of the non-eligible load.  Recovery therefore scales with q.
     The former calculation based on mean eligible power is retained only as
     an explicitly labelled conservative stress scenario.
  4. Peak-cap screening (A8 rule) re-run with c_t -> q * c_t, so the headroom
     effect of q is exact rather than linearised; break-even q against the
     no-flex and 20%-tier baselines.
Outputs: summary.json, f1_K_vs_q.png, f2_net_energy.png, f3_headroom_vs_q.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


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

RERUN = sys.argv[1]   # data/products (public layout) or a run directory with stage subdirs
SYS_CSV = sys.argv[2]
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
QS = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
RHOS = [0.0, 0.5, 1.0]
TS = [0.0, 0.5, 1.0, 4.0]
HS = [1, 4, 24]
ALPHAS = [0.9, 0.95]
N_ALIGN = 60
MARGIN = 0.05
BUDGET_PCT = 0.5


def kval(s, h, a):
    m = sliding_window_view(s, h).min(axis=1) if h > 1 else s
    m = m[~np.isnan(m)]
    return float(np.quantile(m, 1 - a)) if len(m) else np.nan


def feasible(D, cap, s_al, avail, budget_h, L):
    r = np.maximum(0.0, D + L * s_al - cap)
    return bool(np.all(r <= L * avail + 1e-9) and int((r > 0).sum()) <= budget_h)


def max_L(D, cap, s_al, avail, budget_h, hi=20000.0):
    if not feasible(D, cap, s_al, avail, budget_h, 1.0):
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(D, cap, s_al, avail, budget_h, mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    env = pd.read_parquet(_find(RERUN, "a5", "envelope_hourly.parquet"))
    fl = pd.read_parquet(_find(RERUN, "a1", "fleet_hourly_power.parquet"))
    good = env["good"].to_numpy().astype(bool) & fl["have"].to_numpy().astype(bool)
    C = np.where(good, env["curtail_idle_retained"].to_numpy(), np.nan)
    idle_head = np.where(good, env["floor"].to_numpy() + env["shift"].to_numpy(), np.nan)
    mean_C = float(np.nanmean(C))
    out = {"inputs": {"mean_eligible_mw": mean_C, "hours": int(good.sum())},
           "params": {"q": QS, "rho": RHOS, "T_ckpt_h": TS, "note": __doc__.split("Parameters")[1].split("Reported")[0].strip()}}

    # ---- 1. delivered power -----------------------------------------------------
    K = {(h, a): kval(C, h, a) for h in HS for a in ALPHAS}
    homog = {f"h{h}_a{a}": {"K_mw": K[(h, a)],
                            "K_q_mw": {str(q): kval(q * C, h, a) for q in QS},
                            "homogeneity_max_rel_err": float(max(abs(kval(q * C, h, a) - q * K[(h, a)]) / max(K[(h, a)], 1e-9) for q in QS))}
             for h in HS for a in ALPHAS}
    out["delivered_power"] = homog

    # ---- 2. net energy multiplier ------------------------------------------------
    ne = {}
    for h in HS:
        for rho in RHOS:
            for T in TS:
                ne[f"h{h}_rho{rho}_T{T}"] = {"net_over_gross": 1 - rho * (1 + T / (2 * h)),
                                             "rebound_energy_multiple": rho * (1 + T / (2 * h))}
    out["net_energy"] = ne
    out["net_energy_note"] = ("energy relief during the call is q*K*h; rebound is rho*q*K*(h + T/2). "
                              "For rho=1 the call is energy-negative at every h and T>0: it moves energy, it does not save it. "
                              "A capacity product only needs the power during the call, so this bounds the ENERGY claim, not the MW claim.")

    # ---- 3. recovery time and call spacing --------------------------------------
    # Primary contract-consistent calculation: the interrupted/re-executed work
    # is the product actually sold, q*K(0.95,h), not the mean eligible layer.
    # The rebound absorption rate is fixed at 10% of the mean non-eligible layer,
    # so recovery time decreases linearly with q.
    rec_by_q = {}
    rec_q1 = {}
    rec_stress = {}
    idle_mean = float(np.nanmean(idle_head))
    cap_frac, tag = 0.10, "10pct"
    rate = cap_frac * idle_mean

    def recovery_record(interrupted_mw, h, rho, T):
        E_rb = rho * interrupted_mw * (h + T / 2)
        R = E_rb / max(rate, 1e-9)
        spacing = h + R
        return {"interrupted_power_mw": float(interrupted_mw),
                "rebound_energy_mwh": float(E_rb),
                "rebound_absorption_rate_mw": float(rate),
                f"R_min_h_at_{tag}_headroom": float(R),
                "min_spacing_h": float(spacing),
                "max_calls_per_week": float(168 / spacing)}

    for q in QS:
        qrec = {}
        for h in HS:
            for rho in RHOS:
                for T in TS:
                    key = f"h{h}_rho{rho}_T{T}"
                    qrec[key] = recovery_record(q * K[(h, 0.95)], h, rho, T)
        rec_by_q[str(q)] = qrec
    rec_q1 = rec_by_q["1.0"]

    # Backward-looking stress calculation retained only to make the old 6.86 h
    # result auditable.  It assumes the entire mean eligible layer is interrupted,
    # even though only K(0.95,h) is sold; it is not the contract-consistent headline.
    for h in HS:
        for rho in RHOS:
            for T in TS:
                key = f"h{h}_rho{rho}_T{T}"
                rec_stress[key] = recovery_record(mean_C, h, rho, T)
    out["recovery"] = rec_q1
    out["recovery_by_q_sold_K"] = rec_by_q
    out["recovery_mean_eligible_stress_only"] = rec_stress
    out["recovery_note"] = (f"Primary calculation uses sold K_q(0.95,h)=q*K and a fixed rebound-absorption "
                            f"rate equal to 10% of the mean non-eligible layer ({idle_mean:.2f} MW); "
                            "recovery time scales linearly with q; call spacing changes with q but is not proportional "
                            "to q because it also includes the fixed interruption duration. The mean-eligible calculation "
                            "is retained only as a conservative stress scenario and must not be quoted as the sold-product result.")

    # ---- 4. peak-cap screening vs q ----------------------------------------------
    d = pd.read_csv(SYS_CSV)
    d["demand_mw"] = pd.to_numeric(d["demand_mw"], errors="coerce")
    d["utc"] = pd.to_datetime(d["utc_end"], format="%m/%d/%Y %I:%M:%S %p")
    d = d.sort_values("utc").drop_duplicates("utc")
    full = pd.date_range(d["utc"].min(), d["utc"].max(), freq="h")
    D = d.set_index("utc")["demand_mw"].reindex(full).interpolate(limit=48).bfill().ffill().to_numpy()
    nH = len(D)
    F = fl["base_mw"].to_numpy().astype(float)
    t = np.arange(len(F))
    F = np.interp(t, t[good], F[good])
    Cf = np.interp(t, t[good], env["curtail_idle_retained"].to_numpy()[good])
    cap_n = pd.Series(F).rolling(168, min_periods=24).max().bfill().to_numpy()
    s_prof, c_prof = F / cap_n, Cf / cap_n
    rng = np.random.default_rng(11)
    offs = rng.integers(0, len(s_prof), N_ALIGN)
    Pcap = float(D.max()) * (1 + MARGIN)
    budget_h = BUDGET_PCT / 100 * nH
    tile = lambda x, o: np.resize(np.roll(x, o), nH)
    hd = {}
    for q in QS:
        v = [max_L(D, Pcap, tile(s_prof, o), q * tile(c_prof, o), budget_h) for o in offs]
        hd[str(q)] = {"p50": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
    noflex = float(np.median([np.min((Pcap - D) / np.maximum(tile(s_prof, o), 1e-6)) for o in offs]))
    tier20 = float(np.median([max_L(D, Pcap, tile(s_prof, o), 0.20 * tile(s_prof, o), budget_h) for o in offs]))
    qs = np.array(QS)
    hv = np.array([hd[str(q)]["p50"] for q in QS])
    out["headroom_vs_q"] = {"m": MARGIN, "budget_pct": BUDGET_PCT, "by_q_mw": hd, "no_flex_mw": noflex, "tier20_mw": tier20,
                            "gain_over_noflex_pct": {str(q): (hd[str(q)]["p50"] / noflex - 1) * 100 for q in QS},
                            "q_at_half_of_full_gain": float(np.interp(0.5 * (hv[-1] - noflex) + noflex, hv, qs)),
                            "q_break_even_vs_tier20": float(np.interp(tier20, hv, qs)) if hv[-1] >= tier20 else None}

    # ---- headline table -----------------------------------------------------------
    out["headline"] = {
        "K95_4h_mw_by_q": {str(q): q * K[(4, 0.95)] for q in QS},
        "q_needed_for_K95_4h_ge_1MW": float(1.0 / K[(4, 0.95)]),
        "q_needed_for_K95_4h_ge_1.8MW": float(1.8 / K[(4, 0.95)]),
        "recovery_sold_K_q1_h4_rho1_T1": rec_q1["h4_rho1.0_T1.0"],
        "recovery_mean_eligible_stress_q1_h4_rho1_T1": rec_stress["h4_rho1.0_T1.0"],
        "interpretation": ("K is exactly linear in q, so the surface's SHAPE (duration and aggregation "
                           "dependence) is invariant to the realizable fraction; only the level scales. "
                           "The story survives any q; the absolute MW claim does not."),
    }
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ["headline", "headroom_vs_q"]}, indent=1))

    # ---- figures ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for h, c in zip(HS, ["tab:blue", "tab:green", "tab:red"]):
        ax.plot(qs, [q * K[(h, 0.95)] for q in qs], "o-", color=c, label=f"K(0.95, {h} h)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1 MW reference")
    ax.set(xlabel="realizable fraction q of eligible-workload availability", ylabel="delivered MW during the call",
           title="delivered eligible-workload availability vs realizable fraction (exactly linear in q)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "f1_K_vs_q.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for rho, ls in zip(RHOS, ["-", "--", ":"]):
        for T, c in zip(TS, ["#999", "tab:blue", "tab:orange", "tab:red"]):
            ax.plot(HS, [1 - rho * (1 + T / (2 * h)) for h in HS], ls, color=c, marker="o", ms=3,
                    label=f"ρ={rho}, T={T} h" if rho > 0 or T == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(HS, [str(h) for h in HS])
    ax.set(xlabel="call duration h (hours)", ylabel="net energy saved / gross energy shed",
           title="rebound: how much of the energy relief survives re-execution")
    # Explicitly mark the paper's reference case so it cannot be confused with
    # the neighbouring T=4 h curve.
    ref = 1 - 1.0 * (1 + 1.0 / (2 * 4))
    ax.scatter([4], [ref], s=42, facecolor="white", edgecolor="black", zorder=5)
    ax.annotate("ρ=1, T=1 h, h=4 h: −0.125", xy=(4, ref), xytext=(5.0, ref - 0.22),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(OUT, "f2_net_energy.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(qs, hv, yerr=[hv - [hd[str(q)]["p5"] for q in QS], [hd[str(q)]["p95"] for q in QS] - hv],
                fmt="o-", color="tab:green", capsize=2, label="trace-derived eligibility × q")
    ax.axhline(noflex, color="k", ls=":", lw=1.2, label=f"no flexibility ({noflex:.0f} MW)")
    ax.axhline(tier20, color="tab:orange", ls="--", lw=1.2, label=f"assumed 20% interruptible tier ({tier20:.0f} MW)")
    ax.axhline(2000, color="gray", ls="-.", lw=1, label="2 GW reference campus")
    ax.set(xlabel="realizable fraction q", ylabel="admissible AI-DC load (MW)",
           title=f"peak-cap screening vs q (margin {MARGIN:.0%}, {BUDGET_PCT}% invoked-hours budget)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "f3_headroom_vs_q.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
