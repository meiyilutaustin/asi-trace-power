#!/usr/bin/env python
"""A3 join audit + day~105 attribution (revision 2).

(1) Coverage audit of the pod_spans x job_execution_summary join:
    - by day of first observation, span duration bin, GPU-count bin (all spans);
    - implicit coverage by job_type / priority / gpu_spec: matched GPU-hours
      by label divided by total GPU-hours by label from the aggregate table
      (unmatched spans carry no labels, so this is the only label-level view);
    - checks for a jump in coverage or in the delay distribution around day 105.
(2) D(t) = actual - run-on-arrival counterfactual, split by NVIDIA vs XPU and by
    job_type, to test the alternative explanation that the day~120 XPU-A ramp
    (OSDI'26 Fig. 6) rather than a scheduler policy change drives the regime.
(3) Changepoint on the daily p99 |D| series: single-split least squares
    (binary segmentation, first split) with a moving-block bootstrap CI.
Wording: schedule_delay is scheduling latency (OSDI'26: median 1 s, HP P90
101 s), i.e. observed queueing delay, not demonstrated tolerance.
Inputs: agg/pod_spans_b*.parquet, job_execution_summary, pod_hourly_agg.
"""
import json
import os
import sys
import zlib

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
OUT = os.environ.get("OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a3audit_out"))
CFG = os.environ.get("POWER_CFG", os.path.join(os.path.dirname(__file__), "power_curves.yaml"))
os.makedirs(OUT, exist_ok=True)
EDGE_DAYS = 14
DUR_BINS = [0, 1, 2, 6, 24, 72, 168, 1e9]
GPU_BINS = [0, 1, 2, 8, 16, 64, 1e9]


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def watts_per_gpu_hour(cfg):
    pod = pd.read_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"))
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    p = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
    e = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)["total"]
    pod["is_xpu"] = pod["gpu_spec_public"].astype(str).str.startswith("XPU")
    d = pd.DataFrame({"jt": pod["job_type_public"].astype(str), "xpu": pod["is_xpu"], "p": e, "gh": pod["gpu_hours"]})
    g = d.groupby(["jt", "xpu"]).sum()
    wpg = (g["p"] / g["gh"]).to_dict()
    tot = pod.groupby(["job_type_public", "priority_class"], observed=True)["gpu_hours"].sum()
    tot_spec = pod.groupby("gpu_spec_public", observed=True)["gpu_hours"].sum()
    return wpg, tot, tot_spec


def add_steps(arr, pos, height):
    i = np.floor(pos).astype(int)
    frac = pos - i
    np.add.at(arr, i, height * (1 - frac))
    np.add.at(arr, i + 1, height * frac)


def build_series(df, nt):
    a = np.zeros(nt + 2); c = np.zeros(nt + 2)
    h = df["h"].to_numpy()
    add_steps(a, df["t_min"].to_numpy().astype(float), h)
    add_steps(a, (df["t_max"].to_numpy() + 1).astype(float), -h)
    add_steps(c, df["cf_start"].to_numpy(), h)
    add_steps(c, df["cf_end"].to_numpy(), -h)
    return np.cumsum(a)[:nt], np.cumsum(c)[:nt]


def changepoint(y, rng, nboot=300, block=7):
    """Single least-squares split of a daily series; block-bootstrap CI of the day."""
    y = np.asarray(y, float)
    n = len(y)

    def best_split(v):
        best, bk = np.inf, None
        cs = np.cumsum(v); cs2 = np.cumsum(v ** 2)
        for k in range(5, n - 5):
            s1, s2 = cs[k - 1], cs[-1] - cs[k - 1]
            q1, q2 = cs2[k - 1], cs2[-1] - cs2[k - 1]
            sse = (q1 - s1 ** 2 / k) + (q2 - s2 ** 2 / (n - k))
            if sse < best:
                best, bk = sse, k
        return bk, best

    k0, sse0 = best_split(y)
    sse_none = np.sum((y - y.mean()) ** 2)
    m1, m2 = y[:k0].mean(), y[k0:].mean()
    resid = np.concatenate([y[:k0] - m1, y[k0:] - m2])
    ks = []
    for _ in range(nboot):
        starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        rb = np.concatenate([resid[s:s + block] for s in starts])[:n]
        yb = np.concatenate([np.full(k0, m1), np.full(n - k0, m2)]) + rb
        ks.append(best_split(yb)[0])
    return {"day": int(k0), "mean_before": float(m1), "mean_after": float(m2),
            "ratio_after_over_before": float(m2 / m1) if m1 > 0 else None,
            "sse_reduction_frac": float(1 - sse0 / sse_none),
            "ci90_day": [int(np.percentile(ks, 5)), int(np.percentile(ks, 95))]}


def main():
    cfg = yaml.safe_load(open(CFG))
    wpg, tot_gh, tot_spec = watts_per_gpu_hour(cfg)
    js = pd.read_parquet(os.path.join(DATA, "asi_opensource_job_execution_summary"),
                         columns=["pod_id", "gpu_request", "duration_hours", "schedule_delay_sec",
                                  "job_type_public", "priority_class", "gpu_spec_public"])
    n_js_rows = len(js)
    js = js.drop_duplicates("pod_id")
    js["bucket"] = js["pod_id"].map(lambda s: zlib.crc32(s.encode()) % 16).astype("int8")
    parts, unmatched, n_spans, nt = [], [], 0, 0
    for b in range(16):
        sp = pd.read_parquet(os.path.join(AGG, f"pod_spans_b{b:02d}.parquet"))
        n_spans += len(sp)
        nt = max(nt, int(sp["t_max"].max()) + 1)
        m = sp.merge(js[js["bucket"] == b].drop(columns="bucket"), on="pod_id", how="left", indicator=True)
        m["matched"] = m["_merge"] == "both"
        um = m.loc[~m["matched"], ["t_min", "t_max", "gh"]]
        unmatched.append(um)
        parts.append(m[m["matched"]].drop(columns=["pod_id", "_merge"]))
        log(f"bucket {b}: matched {int(m['matched'].sum())}/{len(sp)}")
    df = pd.concat(parts, ignore_index=True); del parts
    um = pd.concat(unmatched, ignore_index=True); del unmatched
    for c in ["job_type_public", "priority_class", "gpu_spec_public"]:
        df[c] = df[c].astype("category")
    df["matched"] = True; um["matched"] = False
    allsp = pd.concat([df[["t_min", "t_max", "gh", "matched"]], um], ignore_index=True)
    allsp["day"] = allsp["t_min"] // 24
    allsp["span_h"] = allsp["t_max"] - allsp["t_min"] + 1
    allsp["gpus"] = allsp["gh"] / allsp["span_h"]
    log(f"spans {n_spans}, matched {len(df)} ({len(df)/n_spans*100:.1f}% of spans, "
        f"{df['gh'].sum()/allsp['gh'].sum()*100:.1f}% of GPU-hours); js rows {n_js_rows}, unique pods {len(js)}")

    def cov(g):
        return pd.DataFrame({"spans_pct": g["matched"].mean() * 100,
                             "gpu_hours_pct": g.apply(lambda x: x.loc[x.matched, "gh"].sum() / max(x["gh"].sum(), 1e-9) * 100),
                             "n_spans": g.size()})
    by_day = cov(allsp.groupby("day"))
    by_dur = cov(allsp.groupby(pd.cut(allsp["span_h"], DUR_BINS, right=False), observed=True))
    by_gpu = cov(allsp.groupby(pd.cut(allsp["gpus"], GPU_BINS, right=False), observed=True))
    m_gh = df.groupby(["job_type_public", "priority_class"], observed=True)["gh"].sum()
    by_label = pd.DataFrame({"matched_gh": m_gh, "agg_gh": tot_gh}).fillna(0)
    by_label["implicit_coverage_pct"] = by_label["matched_gh"] / by_label["agg_gh"].replace(0, np.nan) * 100
    m_spec = df.groupby("gpu_spec_public", observed=True)["gh"].sum()
    by_spec = pd.DataFrame({"matched_gh": m_spec, "agg_gh": tot_spec}).fillna(0)
    by_spec["implicit_coverage_pct"] = by_spec["matched_gh"] / by_spec["agg_gh"].replace(0, np.nan) * 100
    w100 = by_day.loc[95:104, "gpu_hours_pct"].mean(); w110 = by_day.loc[106:115, "gpu_hours_pct"].mean()
    df["delay_h"] = df["schedule_delay_sec"].fillna(0).clip(lower=0) / 3600
    df["day"] = df["t_min"] // 24
    dl = df.groupby("day").apply(lambda x: pd.Series({
        "gh_share_delay_ge1h": x.loc[x.delay_h >= 1, "gh"].sum() / max(x["gh"].sum(), 1e-9),
        "null_delay_gh_share": x.loc[x.schedule_delay_sec.isna(), "gh"].sum() / max(x["gh"].sum(), 1e-9),
        "median_delay_s": float(np.median(x["schedule_delay_sec"].dropna())) if x["schedule_delay_sec"].notna().any() else np.nan}))

    # ---- D(t) by accelerator family and job type ------------------------------
    df["span_h"] = df["t_max"] - df["t_min"] + 1
    df["h"] = df["gh"] / df["span_h"]
    df["cf_start"] = df["t_min"] - df["delay_h"]
    df["cf_end"] = df["t_max"] + 1 - df["delay_h"]
    censored = (df["t_min"] <= 0) | (df["cf_start"] < 0)
    df = df[~censored]
    df["xpu"] = df["gpu_spec_public"].astype(str).str.startswith("XPU")
    good = np.ones(nt, bool); good[:EDGE_DAYS * 24] = False; good[-24:] = False
    series = {}
    for (jt, xpu), sub in df.groupby(["job_type_public", "xpu"], observed=True):
        w = wpg.get((str(jt), bool(xpu)), np.nanmean(list(wpg.values()))) / 1e6
        a, c = build_series(sub, nt)
        series[(str(jt), bool(xpu))] = (a * w, c * w)
    D_all = sum(a - c for a, c in series.values())
    D_nv = sum(a - c for (jt, x), (a, c) in series.items() if not x)
    D_xpu = sum(a - c for (jt, x), (a, c) in series.items() if x)
    D_type = {}
    for (jt, x), (a, c) in series.items():
        D_type[jt] = D_type.get(jt, 0) + (a - c)
    days = np.arange(nt) // 24

    def daily_p99(D):
        s = pd.DataFrame({"d": days[good], "v": np.abs(D[good])}).groupby("d")["v"].quantile(0.99)
        return s

    p99 = {"all": daily_p99(D_all), "nvidia": daily_p99(D_nv), "xpu": daily_p99(D_xpu)}
    for jt, D in D_type.items():
        p99[f"type:{jt}"] = daily_p99(D)
    rng = np.random.default_rng(3)
    cps = {k: changepoint(v.to_numpy(), rng) for k, v in p99.items() if len(v) > 30}
    for k in cps:
        cps[k]["day"] += int(p99[k].index.min()); cps[k]["ci90_day"] = [d + int(p99[k].index.min()) for d in cps[k]["ci90_day"]]

    summary = {
        "join": {"n_spans": int(n_spans), "n_matched": int(len(df) + censored.sum()), "js_rows": int(n_js_rows),
                 "js_unique_pods": int(len(js)), "span_coverage_pct": float(allsp["matched"].mean() * 100),
                 "gpu_hour_coverage_pct": float(allsp.loc[allsp.matched, "gh"].sum() / allsp["gh"].sum() * 100),
                 "coverage_gh_pct_days95_104": float(w100), "coverage_gh_pct_days106_115": float(w110)},
        "coverage_by_duration_bin": by_dur.reset_index().astype(str).to_dict("records"),
        "coverage_by_gpu_bin": by_gpu.reset_index().astype(str).to_dict("records"),
        "implicit_coverage_by_label": by_label.reset_index().astype(str).to_dict("records"),
        "implicit_coverage_by_spec": by_spec.reset_index().astype(str).to_dict("records"),
        "delay_stats_around_105": {"days95_104": dl.loc[95:104].mean().to_dict(), "days106_115": dl.loc[106:115].mean().to_dict()},
        "changepoints_daily_p99_absD": cps,
        "wording": "schedule_delay = scheduling latency (observed queueing delay), not demonstrated tolerance",
    }
    by_day.to_csv(os.path.join(OUT, "coverage_by_day.csv"))
    dl.to_csv(os.path.join(OUT, "delay_by_day.csv"))
    pd.DataFrame({"t": np.arange(nt), "good": good, "D_all": D_all, "D_nvidia": D_nv, "D_xpu": D_xpu,
                  **{f"D_{k}": v for k, v in D_type.items()}}).to_parquet(os.path.join(OUT, "D_by_family.parquet"), index=False)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# A3 join audit — summary\n\n```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    log(json.dumps({k: v for k, v in summary["join"].items()}, indent=1))
    log(json.dumps(cps, indent=1))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(by_day.index, by_day["gpu_hours_pct"], lw=0.8, label="GPU-hour coverage")
    axes[0].plot(by_day.index, by_day["spans_pct"], lw=0.8, label="span coverage")
    axes[0].axvline(105, color="r", ls="--", lw=1); axes[0].set(ylabel="join coverage (%)"); axes[0].legend(fontsize=8)
    axes[1].plot(dl.index, dl["gh_share_delay_ge1h"] * 100, lw=0.8, label="GPU-hours with delay ≥1 h (%)")
    axes[1].plot(dl.index, dl["null_delay_gh_share"] * 100, lw=0.8, label="GPU-hours with null delay (%)")
    axes[1].axvline(105, color="r", ls="--", lw=1); axes[1].set(ylabel="%"); axes[1].legend(fontsize=8)
    for k, c in [("all", "k"), ("nvidia", "tab:green"), ("xpu", "tab:orange")]:
        axes[2].plot(p99[k].index, p99[k].values, lw=0.8, color=c, label=f"{k} (cp day {cps[k]['day']}, CI {cps[k]['ci90_day']})" if k in cps else k)
    axes[2].axvline(105, color="r", ls="--", lw=1); axes[2].axvline(120, color="tab:orange", ls=":", lw=1, label="day 120 XPU-A ramp (OSDI Fig. 6)")
    axes[2].set(xlabel="day", ylabel="daily p99 |D(t)| (MW)"); axes[2].legend(fontsize=7)
    savefig(fig, "f1_audit_and_changepoint.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for jt, D in D_type.items():
        s = daily_p99(D)
        if s.max() > 0.005:
            ax.plot(s.index, s.values, lw=0.8, label=jt)
    ax.axvline(105, color="r", ls="--", lw=1)
    ax.set(xlabel="day", ylabel="daily p99 |D(t)| (MW)", title="shifting by job type")
    ax.legend(fontsize=8)
    savefig(fig, "f2_D_by_type.png")
    log("DONE a3_join_audit")


if __name__ == "__main__":
    main()
