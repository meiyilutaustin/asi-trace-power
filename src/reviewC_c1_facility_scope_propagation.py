#!/usr/bin/env python
"""C1 (reviewer 2026-09-18 §1, highest priority): propagate the M100 util-dependent
shape correction into the FACILITY-POWER scope share, the 1/4/24 h envelope
P / P^ind, and the three decomposition factors.

B3 (runs/2026-09-18_review-B-experiments) showed the envelope/mean shift ratio of
the *eligible* series is ~1 under the correction, and reported the eligible-MW
shift (training -45%, linear -29%). The appendix then *asserted* that the
facility-power scope share "falls by less than the eligible MW do". This script
computes that number.

Method (SENSITIVITY, same caveat as B3 -- M100 is V100 hardware, not the ASI
fleet; this is not a recalibration):
  - Apply the SAME util-dependent total-GPU-power correction c_pow(u) (M100 July,
    training and linear families) as B3 to:
      (i)  the idle-retained LP-active eligible series (identical to B3), and
      (ii) the GPU term of facility power == every allocated GPU pod's total
           power (idle+active), LP and non-LP alike.
    Host power, unallocated-GPU idle, and PUE (1.2) are left UNCHANGED, exactly
    as the facility model in src/a1_load_reconstruction.compute_power builds them
    (P_facility = PUE*(e_active + e_idlefl + e_host)).
  - Recompute (a) mean facility power; (b) scope = mean eligible / mean {workload,
    IT, facility}; (c) the 1/4/24 h alpha=0.95 envelope P (=P3, honest coincident
    aggregate) and P^ind (=P2, independent-summed) in absolute MW; (d) the three
    factors a_scope / a_persist / a_coinc. Training and linear families each.

Base facility reproduction check: GPU 29.37 / IT 43.60 / facility 52.32 MW with
base params on the local 2026-09-02 agg (matches 08_appendix.tex Table).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(HERE, "..", "2026-09-18_review-B-experiments"))
import powermodel as pm  # noqa: E402
import flex_decomposition as fd  # noqa: E402
import b3_m100_bin_propagation as b3  # noqa: E402  (LP-eligible correction machinery)

A1 = os.path.join(HERE, "..", "2026-09-02_a1-load-reconstruction")
POD = os.path.join(A1, "pod_hourly_agg.parquet")
SRV = os.path.join(A1, "server_hourly_agg.parquet")
CFG = os.path.join(REPO, "configs", "power_curves.yaml")
M100_BINNED = os.path.join(HERE, "..", "2026-09-18_review-B-experiments",
                           "m100_july", "a_curve_binned.csv")
OUT = os.path.join(HERE, "c1_facility_scope_propagation")
os.makedirs(OUT, exist_ok=True)

PHI_METERED = 0.138
ALPHA = 0.95
HORIZONS = (1, 4, 24)
NSHIFT = 200
SEED = 20260918


def facility_components(cfg):
    """Per-hour e_active (allocated-pod GPU power, W) with per-row breakdown, plus
    unallocated GPU idle and host power. Mirrors a1_load_reconstruction."""
    pod = pd.read_parquet(POD)
    srv = pd.read_parquet(SRV)
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    srv["t"] = srv["day"].astype(int) * 24 + srv["hour"].astype(int)
    num = ["gpu_hours", "gpu_hours_null_util", "S0", "S25", "S50", "S75",
           "cpu_req_cores", "cpu_used_cores"]
    pod = (pod.groupby(["t", "cluster_id", "gpu_spec_public", "job_type_public",
                        "state_public"], observed=True, dropna=False)[num]
              .sum().reset_index())
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()

    occ = pod.groupby(["t", "cluster_id", "gpu_spec_public"], observed=True)["gpu_hours"].sum()
    cap = srv.set_index(["t", "cluster_id", "gpu_spec_public"])["gpu_count"]
    idle_gh = (cap - occ.reindex(cap.index, fill_value=0.0)).clip(lower=0)
    idle_t = idle_gh.index.get_level_values("t").to_numpy()
    idle_sc, idle_su = pd.factorize(idle_gh.index.get_level_values("gpu_spec_public"))
    psc, psu = pd.factorize(pod["gpu_spec_public"])
    cap_cores = srv.groupby("t")["cpu_capacity_cores"].sum()
    used_cores = pod[pod["state_public"] != "Standby"].groupby("t")["cpu_used_cores"].sum()

    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    p = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
    parts = pm.row_energy(pod, cfg, p, psc, psu)

    # unallocated GPU idle (W per hour) and host power -- UNCHANGED by correction
    e_idlefl = np.zeros(nt)
    itdp = pm.tdp_of_rows(p, cfg, idle_sc, idle_su)
    np.add.at(e_idlefl, idle_t, idle_gh.to_numpy() * itdp * p["idle_frac"])
    e_host = np.zeros(nt)
    np.add.at(e_host, cap_cores.index.to_numpy(), cap_cores.to_numpy() * p["host_idle"])
    np.add.at(e_host, used_cores.index.to_numpy(),
              used_cores.to_numpy() * (p["host_peak"] - p["host_idle"]))

    # per-row mean SM utilization for the correction interp
    w = pod["gpu_hours"].to_numpy()
    u_row = np.where(w > 0, pod["S0"].to_numpy() / np.maximum(w, 1e-9), 0.0)
    return dict(nt=nt, tpod=tpod, total=parts["total"], u_row=u_row,
                e_idlefl=e_idlefl, e_host=e_host, pue=cfg["pue"]["base"])


def facility_scope(fam, comp, binned, mean_elig_unc, mean_elig_corr):
    """mean facility (corrected/uncorrected) and the three scope shares."""
    uu, cc = b3.make_correction_interp(binned, PHI_METERED, fam)
    corr = np.interp(np.clip(comp["u_row"], 0, 1), uu, cc)
    nt = comp["nt"]
    ea0 = np.zeros(nt); eaC = np.zeros(nt)
    np.add.at(ea0, comp["tpod"], comp["total"])
    np.add.at(eaC, comp["tpod"], comp["total"] * corr)
    good = ea0 > 0
    pue = comp["pue"]

    def stat(ea):
        gpu = float(np.mean(ea[good]) / 1e6)
        it = float(np.mean((ea + comp["e_idlefl"] + comp["e_host"])[good]) / 1e6)
        return gpu, it, pue * it

    gpu0, it0, fac0 = stat(ea0)
    gpuC, itC, facC = stat(eaC)
    return {
        "mean_workload_mw": {"uncorrected": gpu0, "corrected": gpuC,
                             "shift_pct": 100 * (gpuC / gpu0 - 1)},
        "mean_IT_mw": {"uncorrected": it0, "corrected": itC,
                       "shift_pct": 100 * (itC / it0 - 1)},
        "mean_facility_mw": {"uncorrected": fac0, "corrected": facC,
                             "shift_pct": 100 * (facC / fac0 - 1)},
        "scope_share_uncorrected": {"workload": mean_elig_unc / gpu0,
                                    "IT": mean_elig_unc / it0,
                                    "facility": mean_elig_unc / fac0},
        "scope_share_corrected": {"workload": mean_elig_corr / gpuC,
                                  "IT": mean_elig_corr / itC,
                                  "facility": mean_elig_corr / facC},
    }


def envelope_and_factors(fam, binned, comp):
    """Corrected/uncorrected P(=P3), P^ind(=P2) and a_scope/a_persist/a_coinc."""
    C0, A0, CC, AC, good, keep, binding = b3.build_corrected_substrate(fam, PHI_METERED, binned)
    mean_elig_unc = float(np.nanmean(A0))
    mean_elig_corr = float(np.nanmean(AC))

    # mean workload of the SAME 13 clusters (denominator for a_scope P0); use the
    # corrected all-pod facility workload only for the scope-share table above.
    rng = np.random.default_rng(SEED)
    out = {"mean_eligible_mw_uncorrected": mean_elig_unc,
           "mean_eligible_mw_corrected": mean_elig_corr,
           "envelope": {}, "factors": {}}
    for label, C, A in (("uncorrected", C0, A0), ("corrected", CC, AC)):
        env = {}
        for T in HORIZONS:
            rng_p = np.random.default_rng(SEED + T)
            P3 = float(fd.kval(A, T, ALPHA))
            P2 = fd.kval_decorrelated(C, T, ALPHA, rng_p, NSHIFT)
            P1 = float(np.nanmean(A))
            env[f"h{T}"] = {"P_envelope_P3_mw": P3, "P_ind_P2_mw": P2,
                            "P1_mean_mw": P1,
                            "a_persist": P2 / P1, "a_coinc": P3 / P2}
        out["envelope"][label] = env
    return out, mean_elig_unc, mean_elig_corr, keep, binding


def main():
    cfg = yaml.safe_load(open(CFG))
    binned = pd.read_csv(M100_BINNED)
    comp = facility_components(cfg)

    res = {"phi_metered_m100_july": PHI_METERED, "phi_model": b3.PHI_MODEL,
           "alpha": ALPHA, "horizons": list(HORIZONS), "n_shift": NSHIFT,
           "pue": cfg["pue"]["base"],
           "note": ("SENSITIVITY: M100 (V100) util-dependent total-GPU-power "
                    "correction applied to (i) the LP-active eligible series "
                    "(as B3) and (ii) the facility GPU term (all allocated pods, "
                    "LP+non-LP). Host, unallocated idle, PUE unchanged. Not a "
                    "recalibration of the ASI fleet."),
           "families": {}}
    for fam in ("training", "linear"):
        env, me_unc, me_corr, keep, binding = envelope_and_factors(fam, binned, comp)
        scope = facility_scope(fam, comp, binned, me_unc, me_corr)
        res["families"][fam] = {"n_clusters": len(keep),
                                "scope": scope, "envelope_and_factors": env,
                                "binding_hour_utilization": binding}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    print("=== C1 facility-scope propagation ===")
    for fam in ("training", "linear"):
        f = res["families"][fam]
        su, sc = f["scope"]["scope_share_uncorrected"], f["scope"]["scope_share_corrected"]
        elig = f["envelope_and_factors"]
        print(f"\n[{fam}]  eligible {elig['mean_eligible_mw_uncorrected']:.3f} -> "
              f"{elig['mean_eligible_mw_corrected']:.3f} MW  "
              f"({100*(elig['mean_eligible_mw_corrected']/elig['mean_eligible_mw_uncorrected']-1):+.1f}%)")
        print(f"  facility {f['scope']['mean_facility_mw']['uncorrected']:.2f} -> "
              f"{f['scope']['mean_facility_mw']['corrected']:.2f} MW "
              f"({f['scope']['mean_facility_mw']['shift_pct']:+.1f}%)")
        print(f"  scope share (facility): {100*su['facility']:.2f}% -> {100*sc['facility']:.2f}%  "
              f"| workload {100*su['workload']:.2f}%->{100*sc['workload']:.2f}%  "
              f"| IT {100*su['IT']:.2f}%->{100*sc['IT']:.2f}%")
        for T in HORIZONS:
            eu = elig["envelope"]["uncorrected"][f"h{T}"]
            ec = elig["envelope"]["corrected"][f"h{T}"]
            print(f"  T={T:2d}h  P {eu['P_envelope_P3_mw']:.3f}->{ec['P_envelope_P3_mw']:.3f}  "
                  f"P^ind {eu['P_ind_P2_mw']:.3f}->{ec['P_ind_P2_mw']:.3f}  "
                  f"a_persist {eu['a_persist']:.3f}->{ec['a_persist']:.3f}  "
                  f"a_coinc {eu['a_coinc']:.3f}->{ec['a_coinc']:.3f}")
    print("\nwrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
