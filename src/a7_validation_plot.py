#!/usr/bin/env python
"""A7a: external validation of the util->power model against published
measurements (no cluster needed; runs on the laptop venv).

Inputs (data/validation/):
  validation_points.csv  per-GPU / per-node measured power fractions of TDP
  diurnal_profiles.csv   daily-shape metrics from traces, papers, simulations
Our model: P/TDP = idle + (1-idle) * g_type(u); base idle 0.20 (0.15-0.25),
anchored g(1): training 1.00, online 0.55, offline 0.85, default 1.0.
Outputs: f1_anchor_validation.png, f2_diurnal_amplitude.png, summary.json.
"""
import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
IDLE, IDLE_RANGE = 0.20, (0.15, 0.25)
G1 = {"training": 1.00, "online_inference": 0.55, "offline_inference": 0.85}
OUR_DIURNAL_AMPL = 0.073     # A1: daily amplitude / mean (facility, reconstructed)
OUR_PEAK_HOUR = 17


def sat(g):
    return IDLE + (1 - IDLE) * g


def sat_band(g):
    return (IDLE_RANGE[0] + (1 - IDLE_RANGE[0]) * g,
            IDLE_RANGE[1] + (1 - IDLE_RANGE[1]) * g)


def num(x):
    """numeric value or midpoint of 'a-b' range; None otherwise."""
    if x is None:
        return None
    x = x.strip()
    m = re.fullmatch(r"~?(\d*\.?\d+)", x)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"~?(\d*\.?\d+)\s*-\s*(\d*\.?\d+)", x)
    if m:
        return 0.5 * (float(m.group(1)) + float(m.group(2)))
    return None


def category(r):
    w, p = r["workload"].lower(), r["phase"].lower()
    if "facility" in p or "cluster" in p or "server" in p or "fit" in p \
            or "energy" in p or "management" in p or "cold_start" in p \
            or "ratio" in p or "floor" in p or "host" in w or "cpu" in p:
        return None
    if "idle" in p or w == "idle" or "idle" in w:
        if "bare" in p or "deep" in p or "node_idle" in p or w == "idle_fleet":
            return "idle: bare"
        return "idle: allocated"
    if "stress" in w or "burn" in p or "hpl" in p or "compute_phase" in p:
        return "stress / compute phase"
    if "training" in w or "training" in p:
        if "trough" in p or "comm" in p:
            return "training: comm trough"
        return "training: mean"
    if "prefill" in p:
        return "inference: prefill"
    if "decode" in p or "serving_power_floor" in p:
        return "inference: decode-dominated"
    if "diffusion" in p or "diffusion" in w:
        return "inference: diffusion"
    if "offline" in p or "offline" in w:
        return "inference: offline batch"
    if "online" in p or "inference" in w or "serving" in p:
        return "inference: online mixed"
    return None


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "validation_points.csv"))))
    pts = []
    for r in rows:
        v = num(r["power_frac_tdp"])
        cat = category(r)
        if v is None or cat is None or v > 1.3:
            continue
        tag = "EXACT" if r["notes"].startswith("EXACT") else (
            "COMPUTED" if r["notes"].startswith("COMPUTED") else "APPROX")
        gpu = r["gpu"]
        h100 = "H100" in gpu or "H800" in gpu
        pts.append(dict(cat=cat, v=v, tag=tag, gpu=gpu, h100=h100,
                        src=r["source"].split("(")[0].strip()[:24]))
    # NLR decode-dominated points are missing (their prompts are prefill-heavy);
    # ML.ENERGY / Splitwise / 1/W-law supply them.
    order = ["idle: bare", "idle: allocated", "training: comm trough",
             "training: mean", "stress / compute phase",
             "inference: decode-dominated", "inference: online mixed",
             "inference: prefill", "inference: offline batch",
             "inference: diffusion"]
    model = {"idle: allocated": (IDLE, IDLE_RANGE),
             "training: mean": (sat(G1["training"]), sat_band(G1["training"])),
             "stress / compute phase": (sat(1.0), sat_band(1.0)),
             "inference: online mixed": (sat(G1["online_inference"]),
                                         sat_band(G1["online_inference"])),
             "inference: decode-dominated": (sat(G1["online_inference"]),
                                             sat_band(G1["online_inference"])),
             "inference: offline batch": (sat(G1["offline_inference"]),
                                          sat_band(G1["offline_inference"]))}
    summary = {"n_points_used": len(pts), "by_category": {}}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    rng = np.random.default_rng(0)
    for i, cat in enumerate(order):
        sel = [p for p in pts if p["cat"] == cat]
        if not sel:
            continue
        vals = np.array([p["v"] for p in sel])
        summary["by_category"][cat] = {
            "n": len(sel), "min": float(vals.min()), "median": float(np.median(vals)),
            "max": float(vals.max()),
            "model_at_saturation": model.get(cat, (None,))[0]}
        for p in sel:
            x = i + rng.uniform(-0.22, 0.22)
            mk = "o" if p["tag"] == "EXACT" else ("s" if p["tag"] == "COMPUTED" else "^")
            ax.scatter(x, p["v"], marker=mk, s=28,
                       facecolor="tab:blue" if p["h100"] else "none",
                       edgecolor="tab:blue" if p["h100"] else "tab:gray",
                       linewidths=1, alpha=0.85, zorder=3)
        if cat in model:
            m, (lo, hi) = model[cat]
            ax.plot([i - 0.35, i + 0.35], [m, m], color="tab:red", lw=2, zorder=4)
            ax.fill_between([i - 0.35, i + 0.35], lo, hi, color="tab:red",
                            alpha=0.15, zorder=2)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("GPU power / TDP")
    ax.set_title("util→power model anchors vs published measurements "
                 "(red = model at saturation u→1, band = idle-fraction range)")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", color="tab:blue", ls="", label="exact (text/table/data)"),
           Line2D([], [], marker="s", color="tab:blue", ls="", label="computed from raw dataset (NLR)"),
           Line2D([], [], marker="^", color="tab:blue", ls="", label="read from figure"),
           Line2D([], [], marker="o", markerfacecolor="none", color="tab:gray", ls="",
                  label="non-H100 GPU (hollow)"),
           Line2D([], [], color="tab:red", lw=2, label="our anchored model")]
    ax.legend(handles=leg, fontsize=7.5, loc="upper left")
    fig.savefig(os.path.join(OUT, "f1_anchor_validation.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- diurnal amplitude comparison ---------------------------------------
    drows = list(csv.DictReader(open(os.path.join(DATA, "diurnal_profiles.csv"))))
    bars = [("this work: reconstructed facility power\n(155k GPUs, 6 months)",
             OUR_DIURNAL_AMPL, "measured-derived power")]
    for r in drows:
        m = r["metric"].lower()
        if "amplitude" not in m or "normalized" in m:
            continue
        v = num(r["value"])
        if v is None:
            continue
        src, fac = r["source"], r["facility"]
        if "Azure" in src and "requests" in m:
            kind = "conv" if "conv" in fac.lower() else "code"
            bars.append((f"Azure LLM trace 2024 ({kind}): requests", v, "request demand"))
        elif "BurstGPT" in src and "requests" in m:
            bars.append(("BurstGPT: requests", v, "request demand"))
        elif "Vercellino" in src:
            bars.append((f"NLR simulated {fac[:11]}", v, "simulated power"))
    # Helios utilisation night dip 5-8%, Google campus shaping 1-2% (text)
    bars.append(("Helios (SenseTime): GPU allocation util night dip", 0.065, "measured utilisation"))
    bars.append(("Google campus: carbon-aware shaping swing", 0.015, "measured power"))
    # de-duplicate NLR simulated entries: keep min/max as two bars
    nlr = [b for b in bars if b[2] == "simulated power"]
    bars = [b for b in bars if b[2] != "simulated power"]
    if nlr:
        vs = [b[1] for b in nlr]
        bars.append((f"NLR simulated facilities (n={len(nlr)}, min)", min(vs), "simulated power"))
        bars.append((f"NLR simulated facilities (max)", max(vs), "simulated power"))
    cols = {"measured-derived power": "tab:green", "measured power": "tab:green",
            "measured utilisation": "tab:olive", "request demand": "tab:orange",
            "simulated power": "tab:gray"}
    bars.sort(key=lambda b: b[1])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], color=[cols[b[2]] for b in bars])
    ax.set_yticks(y, [b[0] for b in bars], fontsize=7.5)
    ax.set_xlabel("daily amplitude (peak − trough) / mean")
    ax.set_xscale("log")
    ax.set_title("daily amplitude: power vs request demand vs simulation")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:green", label="measured / reconstructed power"),
                       Patch(color="tab:olive", label="measured utilisation"),
                       Patch(color="tab:orange", label="request demand (not power)"),
                       Patch(color="tab:gray", label="simulated facility power")],
              fontsize=7.5, loc="lower right")
    fig.savefig(os.path.join(OUT, "f2_diurnal_amplitude.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    summary["diurnal_bars"] = [{"label": b[0].replace("\n", " "), "amplitude": b[1],
                                "kind": b[2]} for b in bars]
    json.dump(summary, open(os.path.join(OUT, "summary_a7a.json"), "w"), indent=2)
    print(json.dumps(summary["by_category"], indent=1))


if __name__ == "__main__":
    main()
