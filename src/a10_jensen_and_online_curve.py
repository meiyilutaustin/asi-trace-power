#!/usr/bin/env python
"""Pre-writing items 3 and 6 (local, venv).

(3) Jensen / aggregation bias bound.  The power model applies a piecewise-linear
    curve g(u) to the hourly-mean SM utilisation.  For a within-hour utilisation
    distribution with mean m on [0, 1], the exact hourly energy is E[g(u)]; the
    model uses g(m).  Because g is piecewise-linear, |E[g(u)] - g(m)| is bounded
    by the two-point distributions on the knot grid.  We report, per curve
    (training / online with floor / offline / linear):
      - worst-case relative error of hourly energy vs mean utilisation m
        (max over all two-point mixtures with mean m, support in [0,1]);
      - the error for a "duty-cycle" pattern (u alternates between 0 and u_max)
        typical of training comm/compute phases;
      - the fleet-level consequence: a fleet-wide bias band on GPU pod power if
        every hour had the worst-case pattern (upper bound) and the duty-cycle
        pattern with u_max = 2m (typical).
    Note: the linear (FWB) family has zero Jensen error by construction, and the
    A1 Monte Carlo already mixes the linear and anchored families 50/50.

(6) Online-inference power vs SM utilisation: our v2 curve (floor c0 + affine
    compression) against measured decode-dominated / prefill-heavy points
    (validation_points.csv), showing where the v1 curve was low.
Outputs: jensen_bounds.json, f1_jensen_bound.png, f2_online_curve_vs_measured.png
"""
import csv
import json
import os
import re
import sys

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = sys.argv[1]
DATA = sys.argv[2]
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
KNOTS = np.array([0, .25, .5, .75, 1.0])


def g_of(knots_vals):
    kv = np.asarray(knots_vals, float)
    return lambda u: np.interp(u, KNOTS, kv)


def power_frac(g, u, phi, c0=0.0, g1=1.0):
    """P/TDP for util u given curve g; online floor via affine compression."""
    ge = g(u)
    if c0 > 0:
        ge = c0 + (1 - c0 / g1) * ge
    return phi + (1 - phi) * ge


def worst_case_gap(pf, m, n=400):
    """max |E[pf(u)] - pf(m)| over two-point mixtures {a, b} with mean m."""
    best = 0.0
    a_grid = np.linspace(0, m, n)
    for a in a_grid:
        for b in np.linspace(m, 1, 40):
            if b - a < 1e-9:
                continue
            w = (m - a) / (b - a)          # weight on b
            e = (1 - w) * pf(a) + w * pf(b)
            best = max(best, abs(e - pf(m)))
    return best


def main():
    cfg = yaml.safe_load(open(CFG))
    phi = cfg["idle_frac"]["base"]
    floor = cfg["online_floor_frac"]["base"]
    c0 = (floor - phi) / (1 - phi)
    fam = cfg["curve_families"]["anchored"]
    curves = {
        "training": (g_of(fam["training"]), 0.0, 1.0),
        "online_inference (v2, floor)": (g_of(fam["online_inference"]), c0, float(fam["online_inference"][-1])),
        "offline_inference": (g_of(fam["offline_inference"]), 0.0, 1.0),
        "linear (FWB)": (g_of([0, .25, .5, .75, 1]), 0.0, 1.0),
    }
    ms = np.linspace(0.02, 0.98, 25)
    out = {"phi": phi, "online_floor": floor, "c0": c0, "curves": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (g, cc, g1) in curves.items():
        pf = lambda u, g=g, cc=cc, g1=g1: power_frac(g, u, phi, cc, g1)
        worst = np.array([worst_case_gap(pf, m) / pf(m) for m in ms])
        duty = []
        for m in ms:
            umax = min(1.0, 2 * m)
            w = m / umax
            e = (1 - w) * pf(0.0) + w * pf(umax)
            duty.append((e - pf(m)) / pf(m))
        duty = np.array(duty)
        out["curves"][name] = {"worst_case_rel_error_max": float(worst.max()),
                               "worst_case_rel_error_at_m0.3": float(np.interp(0.3, ms, worst)),
                               "duty_cycle_rel_error_at_m0.3": float(np.interp(0.3, ms, duty)),
                               "duty_cycle_rel_error_at_m0.6": float(np.interp(0.6, ms, duty)),
                               "sign_duty_cycle": "model too low (curve convex)" if np.mean(duty) > 0 else "model too high (curve concave)"}
        axes[0].plot(ms, worst * 100, label=name)
        axes[1].plot(ms, duty * 100, label=name)
    axes[0].set(xlabel="hourly-mean SM utilisation m", ylabel="worst-case |E[g(u)] − g(m)| / g(m) (%)",
                title="Jensen bound: any within-hour distribution")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(xlabel="hourly-mean SM utilisation m", ylabel="error for 0/2m duty cycle (%)",
                title="duty-cycle pattern (comm/compute alternation)")
    axes[0].legend(fontsize=7.5); axes[1].legend(fontsize=7.5)
    fig.savefig(os.path.join(OUT, "f1_jensen_bound.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    json.dump(out, open(os.path.join(OUT, "jensen_bounds.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))

    # ---- (6) online curve vs measured ----------------------------------------
    rows = list(csv.DictReader(open(os.path.join(DATA, "validation_points.csv"))))
    pts = []
    for r in rows:
        p = r["phase"].lower(); w = r["workload"].lower()
        if not ("online" in p or "decode" in p or "serving" in p or "prefill" in p or ("inference" in w and "diffusion" not in w and "offline" not in p)):
            continue
        m = re.fullmatch(r"~?(\d*\.?\d+)(?:\s*-\s*(\d*\.?\d+))?", r["power_frac_tdp"].strip())
        if not m:
            continue
        v = float(m.group(1)) if not m.group(2) else 0.5 * (float(m.group(1)) + float(m.group(2)))
        kind = "decode-dominated" if ("decode" in p or "serving_power_floor" in p) else ("prefill" if "prefill" in p else "online mixed")
        pts.append((kind, v, r["source"].split("(")[0].strip()[:20]))
    u = np.linspace(0, 1, 101)
    g_on = g_of(fam["online_inference"]); g1 = float(fam["online_inference"][-1])
    v1 = power_frac(g_on, u, phi)
    v2 = power_frac(g_on, u, phi, c0, g1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(u, v1, "k--", lw=1.2, label="v1 curve (no floor): 0.23 TDP at u=0.05")
    ax.plot(u, v2, "tab:red", lw=2, label=f"v2 curve: floor {floor:.2f} TDP, saturation {v2[-1]:.2f}")
    ax.fill_between(u, power_frac(g_on, u, 0.17, (0.35 - 0.17) / 0.83, g1), power_frac(g_on, u, 0.21, (0.50 - 0.21) / 0.79, g1),
                    color="tab:red", alpha=0.15, label="MC range (floor 0.35–0.50, idle 0.17–0.21)")
    ax.axvspan(0.02, 0.10, color="tab:blue", alpha=0.08, label="OSDI'26: GenAI serving median SM util 5–6%")
    for kind, mk, c in [("decode-dominated", "o", "tab:blue"), ("online mixed", "s", "tab:gray"), ("prefill", "^", "tab:orange")]:
        vals = [v for k, v, _ in pts if k == kind]
        if vals:
            ax.scatter(np.full(len(vals), {"decode-dominated": 0.06, "online mixed": 0.5, "prefill": 0.9}[kind]) + np.random.default_rng(0).uniform(-0.02, 0.02, len(vals)),
                       vals, marker=mk, color=c, s=30, label=f"measured: {kind} (n={len(vals)}, x = assumed regime)")
    ax.set(xlabel="SM utilisation u", ylabel="GPU power / TDP", ylim=(0, 1.05),
           title="online-inference power model v1 vs v2 against measured serving power")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(os.path.join(OUT, "f2_online_curve_vs_measured.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("points:", len(pts))


if __name__ == "__main__":
    main()
