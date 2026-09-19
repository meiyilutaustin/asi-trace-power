#!/usr/bin/env python
"""B3 (reviewer 2026-09-18 §1): utilization-resolved M100 model error and its
propagation into the 1/4/24 h flexibility envelope.

The prior M100 validation reported only the AGGREGATE energy bias (training curve
+72.7% high on hourly means). The reviewer asks for (i) low-utilization-bin error
and (ii) the impact of that util-dependent error on the low-tail envelope, which
is set by low-load (low-utilization) hours and could therefore be biased more
than the aggregate energy suggests.

Part A - util-resolved error on M100 July metered GPUs:
  For each hourly-mean utilization bin, the multiplicative error of the paper's
  curve g relative to the M100 metered curve, c_g(u) = g_metered(u)/g_model(u),
  for the training-anchored and linear families (the two the power-MC mixes).
  g_metered(u) = (metered_power/TDP - phi_metered)/(1-phi_metered).

Part B - envelope propagation (SENSITIVITY, clearly labelled):
  M100 is V100 hardware, NOT the ASI fleet, so this is not a recalibration of the
  ASI clusters. It asks: if the ASI reconstruction carried a utilization-dependent
  shape error of the magnitude measured on external metered hardware, how much
  would the 1/4/24 h envelope move, and does it move MORE than the fleet mean
  (which is all the aggregate energy check constrains)?
  We apply c_g(u_row) (u_row = S0/gpu_hours, the row's mean SM utilization) to the
  curtailable active energy of every LP row, rebuild the aggregate eligible series,
  and recompute P(T,0.95). We report corrected/uncorrected for the mean and for
  the 1/4/24 h envelope, plus the utilization composition of the binding hours.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import powermodel as pm  # noqa: E402
import flex_decomposition as fd  # noqa: E402

AGG = os.path.join(HERE, "..", "2026-09-02_a1-load-reconstruction", "pod_hourly_agg.parquet")
CFG = os.path.join(HERE, "..", "..", "configs", "power_curves.yaml")
M100_BINNED = os.path.join(HERE, "m100_july", "a_curve_binned.csv")
OUT = os.path.join(HERE, "b3_m100_bin_propagation")
os.makedirs(OUT, exist_ok=True)

KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CURVES = {"training": [0, 0.50, 0.75, 0.90, 1.00],
          "linear": [0, 0.25, 0.50, 0.75, 1.00]}
PHI_MODEL = 0.20
ALPHA = 0.95
HORIZONS = (1, 4, 24)


def g_of_u(kv, u):
    return np.interp(np.clip(u, 0, 1), KNOTS, kv)


def build_metered_g(binned, phi_metered):
    """Empirical g_metered(u) from the M100 binned metered power/TDP."""
    u = binned["u_mean"].to_numpy()
    p_over_tdp = binned["p_over_tdp"].to_numpy()
    g_met = np.clip((p_over_tdp - phi_metered) / (1 - phi_metered), 0, None)
    return u, g_met


def part_a(binned, phi_metered):
    u, g_met = build_metered_g(binned, phi_metered)
    rows = []
    for fam in CURVES:
        g_mod = g_of_u(CURVES[fam], u)
        # correction on g (active part) and on total power (phi+(1-phi)g)
        c_g = np.where(g_mod > 1e-9, g_met / g_mod, np.nan)
        p_met = phi_metered + (1 - phi_metered) * g_met
        p_mod = PHI_MODEL + (1 - PHI_MODEL) * g_mod
        for i in range(len(u)):
            rows.append({"family": fam, "u_lo": float(binned["u_lo"].iloc[i]),
                         "u_hi": float(binned["u_hi"].iloc[i]),
                         "u_mean": float(u[i]), "n_gpu_hours": int(binned["n_gpu_hours"].iloc[i]),
                         "g_metered": float(g_met[i]), "g_model": float(g_mod[i]),
                         "active_model_over_metered": float(g_mod[i] / g_met[i]) if g_met[i] > 1e-9 else None,
                         "power_model_over_metered_pct": float(100 * (p_mod[i] / p_met[i] - 1))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a_util_resolved_error.csv"), index=False)
    # highlight low-utilization bins
    low = {fam: {} for fam in CURVES}
    for fam in CURVES:
        d = df[df.family == fam]
        for lab, hi in (("u_lt_0.1", 0.1), ("u_lt_0.3", 0.3)):
            sub = d[d.u_hi <= hi]
            if len(sub):
                w = sub["n_gpu_hours"].to_numpy()
                low[fam][lab] = {
                    "n_gpu_hours": int(w.sum()),
                    "mean_power_model_over_metered_pct": float(
                        np.average(sub["power_model_over_metered_pct"], weights=w))}
    return df, low


def make_correction_interp(binned, phi_metered, family):
    """c_pow(u) interpolator: metered/model TOTAL GPU power ratio, from M100 bins.
    Total power (phi+(1-phi)g) is what M100 meters and is well-posed at u->0
    (unlike the active-only g ratio, which divides by g_model~0 at idle)."""
    u, g_met = build_metered_g(binned, phi_metered)
    g_mod = g_of_u(CURVES[family], u)
    p_met = phi_metered + (1 - phi_metered) * g_met
    p_mod = PHI_MODEL + (1 - PHI_MODEL) * g_mod
    c = p_met / p_mod                                    # bounded ~0.5..0.8
    # extend flat to the ends
    uu = np.concatenate([[0.0], u, [1.0]])
    cc = np.concatenate([[c[0]], c, [c[-1]]])
    order = np.argsort(uu)
    return uu[order], cc[order]


def build_corrected_substrate(family, phi_metered, binned):
    """Rebuild the eligible curtailable series C with c_g(u_row) applied to LP
    active energy; return uncorrected/corrected C, A and per-row diagnostics."""
    cfg = yaml.safe_load(open(CFG))
    boundary = cfg.get("curtail_boundary", "idle_retained")
    elig = cfg.get("eligibility", "central")
    pod = pd.read_parquet(AGG)
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    cl_codes, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
    ncl = len(cl_uniq)
    layer = pm.assign_layer(pod, elig)
    p0 = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
    parts = pm.row_energy(pod, cfg, p0, spec_codes, spec_uniq)
    curt_e = pm.curtail_energy(parts, boundary)          # active (Wh) per row

    # row mean SM utilization u_row = S0 / gpu_hours (S0 = sum w*u)
    w = pod["gpu_hours"].to_numpy()
    u_row = np.where(w > 0, pod["S0"].to_numpy() / np.maximum(w, 1e-9), 0.0)
    uu, cc = make_correction_interp(binned, phi_metered, family)
    corr = np.interp(np.clip(u_row, 0, 1), uu, cc)

    m = layer == "curtail"
    curt0 = np.zeros((ncl, nt)); curtC = np.zeros((ncl, nt)); work = np.zeros((ncl, nt))
    np.add.at(curt0, (cl_codes[m], tpod[m]), curt_e[m] / 1e6)
    np.add.at(curtC, (cl_codes[m], tpod[m]), (curt_e[m] * corr[m]) / 1e6)
    np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
    good = work.sum(axis=0) > 0

    order = np.argsort(-curt0.mean(axis=1))
    keep = [i for i in order if curt0[i].mean() >= fd.MIN_CLUSTER_MW]
    def series(curt):
        C = curt[keep].astype(float); C[:, ~good] = np.nan
        A = np.nansum(C, axis=0); A[~good] = np.nan
        return C, A
    C0, A0 = series(curt0)
    CC, AC = series(curtC)

    # utilization composition of the binding hours of each horizon (uncorrected)
    binding = {}
    for T in HORIZONS:
        from numpy.lib.stride_tricks import sliding_window_view
        if T == 1:
            rmv = A0.copy(); starts = np.arange(nt)
        else:
            rmv = sliding_window_view(A0, T).min(axis=1)
            starts = np.arange(len(rmv))
        okm = ~np.isnan(rmv)
        rmv2 = rmv[okm]; starts2 = starts[okm]
        thr = np.quantile(rmv2, 1 - ALPHA)
        # the windows at/below the (1-alpha) quantile are the binding ones
        binds = starts2[rmv2 <= thr]
        hrs = np.unique(np.concatenate([np.arange(s, s + T) for s in binds])) if len(binds) else np.array([], int)
        hrs = hrs[(hrs >= 0) & (hrs < nt)]
        # mean curtail-row utilization in those hours, gpu-hour weighted
        hmask = np.isin(tpod, hrs) & m
        ww = w[hmask]
        binding[f"h{T}"] = {"n_binding_hours": int(len(hrs)),
                            "mean_curtail_row_util": float(np.average(u_row[hmask], weights=ww)) if ww.sum() else None}
    return C0, A0, CC, AC, good, keep, binding


def part_b(binned):
    phi_metered = 0.138   # M100 idle_over_tdp_median (July)
    res = {}
    for family in CURVES:
        C0, A0, CC, AC, good, keep, binding = build_corrected_substrate(family, phi_metered, binned)
        mean0 = float(np.nanmean(A0)); meanC = float(np.nanmean(AC))
        env0 = {T: float(fd.kval(A0, T, ALPHA)) for T in HORIZONS}
        envC = {T: float(fd.kval(AC, T, ALPHA)) for T in HORIZONS}
        res[family] = {
            "n_clusters": len(keep),
            "mean_eligible_mw_uncorrected": mean0,
            "mean_eligible_mw_corrected": meanC,
            "mean_shift_pct": 100 * (meanC / mean0 - 1),
            "envelope_mw_uncorrected": {f"h{T}": env0[T] for T in HORIZONS},
            "envelope_mw_corrected": {f"h{T}": envC[T] for T in HORIZONS},
            "envelope_shift_pct": {f"h{T}": 100 * (envC[T] / env0[T] - 1) for T in HORIZONS},
            "envelope_shift_over_mean_shift": {
                f"h{T}": (100 * (envC[T] / env0[T] - 1)) / (100 * (meanC / mean0 - 1))
                if abs(meanC / mean0 - 1) > 1e-9 else None for T in HORIZONS},
            "binding_hour_utilization": binding,
        }
    return res


def main():
    binned = pd.read_csv(M100_BINNED)
    phi_metered = 0.138
    dfa, low = part_a(binned, phi_metered)
    b = part_b(binned)
    out = {"phi_metered_m100_july": phi_metered, "phi_model": PHI_MODEL,
           "alpha": ALPHA, "horizons": list(HORIZONS),
           "note": ("Part A: util-resolved model/metered error on M100 V100 GPUs. "
                    "Part B: SENSITIVITY of the ASI envelope to a util-dependent "
                    "shape error of that magnitude (M100 is not the ASI fleet; not "
                    "a recalibration). envelope_shift_over_mean_shift > 1 means the "
                    "low-tail envelope is biased MORE than the fleet mean, i.e. the "
                    "aggregate energy check alone under-states the low-tail risk."),
           "part_a_low_util_error": low,
           "part_b_envelope_propagation": b}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print("PART A low-utilization model-over-metered power error (weighted):")
    for fam in low:
        print(f"  {fam}: {low[fam]}")
    print("\nPART B envelope propagation (corrected/uncorrected):")
    for fam in b:
        r = b[fam]
        print(f"  [{fam}] mean shift {r['mean_shift_pct']:.1f}%  "
              f"envelope shift {{'h1':{r['envelope_shift_pct']['h1']:.1f}, "
              f"'h4':{r['envelope_shift_pct']['h4']:.1f}, 'h24':{r['envelope_shift_pct']['h24']:.1f}}}%  "
              f"ratio env/mean { {k: round(v,2) for k,v in r['envelope_shift_over_mean_shift'].items()} }")
        print(f"        binding-hour mean curtail util: { {k: round(v['mean_curtail_row_util'],3) for k,v in r['binding_hour_utilization'].items()} }")
    print("\nwrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
