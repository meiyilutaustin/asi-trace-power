#!/usr/bin/env python
"""Pre-writing item 9: within-hour power minima from NREL 5 Hz per-GPU logs.

For every complete 60-min window (sliding, 10-min step) in each NREL run, compute
   r_1min  = min(1-min mean power) / hour mean power
   r_5min  = min(5-min mean power) / hour mean power
   r_15min = min(15-min mean power) / hour mean power
per GPU, by workload class.  This answers "an hourly average does not prove
continuous availability": the ratio tells how far below the hourly mean the
short-interval power dips.  Applied to the datacenter's eligible layer it is a
haircut on K for products settled at 5/15-min intervals (upper bound on the
haircut, because it comes from single-node benchmark runs, not a fleet).
Usage: a11_intrahour_minima.py <nlr_312_root> <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FMT = "%Y-%m-%d_%H:%M:%S.%f"
STEPS = {"1min": "1min", "5min": "5min", "15min": "15min"}


def read_nvml(p):
    hdr = None
    with open(p) as f:
        for line in f:
            if line.startswith("#") and "timestamp" in line:
                hdr = line.replace("#", "").split(); break
    df = pd.read_csv(p, sep=r"\s+", comment="#", header=None)
    df.columns = hdr[:len(df.columns)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=FMT)
    df = df.set_index("timestamp")
    cols = [c for c in df.columns if c.endswith("[mW]")]
    g = df[cols] / 1000.0
    g.columns = [c.replace("[mW]", "") for c in cols]
    return g


def windows(g, cls, rows):
    if len(g) < 100:
        return
    span_h = (g.index[-1] - g.index[0]).total_seconds() / 3600
    if span_h < 1.0:
        return
    mins = {k: g.resample(v).mean() for k, v in STEPS.items()}
    t0 = g.index[0].ceil("10min")
    while t0 + pd.Timedelta(hours=1) <= g.index[-1]:
        t1 = t0 + pd.Timedelta(hours=1)
        w = g[(g.index >= t0) & (g.index < t1)]
        if len(w) > 0.8 * 3600 * 5 * 0.5:      # at least ~40% of 5 Hz samples
            hm = w.mean()
            rec = {"class": cls, "t0": str(t0), "hour_mean_w_pergpu": float(hm.mean()), "n_gpus": int(w.shape[1])}
            for k in STEPS:
                m = mins[k][(mins[k].index >= t0) & (mins[k].index < t1)]
                rec[f"r_{k}"] = float((m.min() / hm).mean())       # per-GPU ratio, averaged over GPUs
                rec[f"r_{k}_node"] = float(m.sum(axis=1).min() / hm.sum())
            rows.append(rec)
        t0 += pd.Timedelta(minutes=10)


def main():
    rows = []
    groups = {"training_llama2_70b_lora": "training (LoRA)", "training_stable_diffusion": "training (SD)",
              "inference_online_rate_llama3_70b": "online inference (rate sweep)",
              "inference_online_finite_llama3_70b": "online inference (finite)",
              "inference_offline_llama3_70b": "offline inference"}
    for d, cls in groups.items():
        for p in sorted(glob.glob(os.path.join(ROOT, "00_raw_datasets", d, "**", "nvml_*.log"), recursive=True)):
            try:
                g = read_nvml(p)
            except Exception as e:
                print("skip", p, e); continue
            windows(g, cls, rows)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "intrahour_windows.csv"), index=False)
    summ = {}
    for cls, sub in df.groupby("class"):
        summ[cls] = {"n_windows": int(len(sub)), "hour_mean_frac_tdp": float(sub.hour_mean_w_pergpu.mean() / 700)}
        for k in STEPS:
            summ[cls][f"r_{k}_median"] = float(sub[f"r_{k}"].median())
            summ[cls][f"r_{k}_p10"] = float(sub[f"r_{k}"].quantile(0.1))
            summ[cls][f"r_{k}_node_p10"] = float(sub[f"r_{k}_node"].quantile(0.1))
    summ["_note"] = ("ratio = min short-interval mean / hourly mean per GPU (node = 4-GPU sum). "
                     "Single-node benchmark runs (NLR 2604.07345 raw NVML, 5 Hz); fleet aggregation would raise the ratios.")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    classes = [c for c in summ if not c.startswith("_")]
    x = np.arange(len(classes))
    for j, k in enumerate(STEPS):
        ax.bar(x + (j - 1) * 0.27, [summ[c][f"r_{k}_p10"] for c in classes], 0.27, label=f"P10 of min {k} mean / hour mean")
    ax.set_xticks(x, classes, rotation=15, fontsize=8)
    ax.set(ylabel="ratio", ylim=(0, 1.05), title="within-hour power minima relative to the hourly mean (NLR H100 node runs)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "f1_intrahour_minima.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
