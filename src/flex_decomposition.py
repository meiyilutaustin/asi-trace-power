#!/usr/bin/env python3
"""Flexibility-overstatement decomposition (SPEC 2026-09-15, strong version).

Re-analysis on EXISTING model outputs/series (E0-E4). No new power-model fit, no
new simulation, no scheduler. Turns the three phenomena (scope / persistence /
coincidence) into a strictly-nested multiplicative decomposition of how much the
fixed-share + site-independence planning assumptions overstate firm curtailable
power, with an exact identity, order-invariant (Shapley) attribution, and
moving-block bootstrap CIs.

Substrate (E0): the 13 Alibaba internal clusters, idle-retained LP-active
eligibility, GPU-side workload MW basis (== facility-marginal at multiplier 1.0,
same convention as const_over_obs). Firm curtailable power is

    P(x, T, alpha) = (1 - alpha) quantile of the T-hour running minimum of x.

Nested planners, one assumption relaxed per step:
  P0  straw man        s0 * mean facility(GPU-side) demand, flat, independent
  P1  scope-corrected  measured mean eligible level  = sum_i mean(C_i) = mean(A)
  P2  persist-corrected sum_i P(C_i,T,a)             (still independent-summed)
  P3  coincidence(honest) P(A,T,a)                    (actual coincident aggregate)
Factors: a_scope = P1/P0, a_persist(T)=P2/P1, a_coinc(T)=P3/P2.
Identity by construction (exact): P3 = P0 * a_scope * a_persist * a_coinc.
The identity is trivially exact; the contribution is the MAGNITUDE of each factor.

E2 Shapley: a cube value function over {scope, dur, coinc}; scope is a pure level
scalar (s0/s* when off), dur toggles running-min vs mean, coinc toggles
aggregate-first vs per-site-then-sum. Shapley = mean marginal log-contribution
over all 3!=6 relaxation orders; shares sum exactly to log(P3/P0). The per-factor
SPREAD across the 6 orders is the readout: small -> near-independent multiplicative
mechanisms; large -> the effects interact (Shapley is then the honest headline).

E3: joint moving-block bootstrap over time (blocks 168/336 h, 1000 rep), NaN at
block splices so windows never straddle; propagated to every factor and Shapley
share. Preprocessing/masks fixed -> intervals are conditional.

E4: (a) power-MC ensemble structure test (each factor stays <1 and monotone in T?);
(b) a_coinc only on external panels (ASI / Helios) as STRUCTURE replication;
(c) alpha in {0.90,0.95,0.99} and the s0 sweep.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import powermodel as pm  # noqa: E402
from provenance import provenance  # noqa: E402

HORIZONS = (1, 4, 24)
ALPHAS = (0.90, 0.95, 0.99)
ALPHA_PRIMARY = 0.95
MIN_CLUSTER_MW = 0.02              # same cohort rule as A6 (a6_accreditation.py)
BLOCKS = (168, 336)
ASSUMPTIONS = ("scope", "dur", "coinc")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rolling_min(series, h):
    if h == 1:
        m = np.asarray(series, float)
    else:
        m = sliding_window_view(np.asarray(series, float), h).min(axis=1)
    return m[~np.isnan(m)]


def kval(series, h, alpha):
    """P(x,T,alpha): (1-alpha) quantile of the h-hour running minimum."""
    m = rolling_min(series, h)
    return float(np.quantile(m, 1 - alpha)) if len(m) else np.nan


TREND_H = 168


def trend_of(mat):
    """168 h centred rolling mean per cluster (NaN-aware); a6_accreditation.trend_of."""
    return np.vstack([pd.Series(r).rolling(TREND_H, center=True, min_periods=24).mean()
                      .bfill().ffill().to_numpy() for r in mat])


def kval_decorrelated(C, h, alpha, rng, n_shift):
    """Independent-across-sites firm power: detrend, circularly shift each site's
    residual (removing cross-cluster covariance while keeping each site's growth
    phase and marginal), re-add its trend, sum, then P(.,h,alpha). Averaged over
    n_shift draws. This is the 'independent-summed' planner (a6 null_indep_v2)."""
    tr = trend_of(C)
    res = C - tr
    nt = C.shape[1]
    vals = []
    for _ in range(n_shift):
        s = np.zeros(nt)
        for i in range(C.shape[0]):
            s += np.roll(res[i], rng.integers(0, nt)) + tr[i]
        vals.append(kval(np.maximum(s, 0.0), h, alpha))
    return float(np.nanmean(vals))


# --------------------------------------------------------------------------- E0
def build_substrate(agg_path, cfg_path):
    """Per-cluster eligible-curtailable series C (N x nt), aggregate A, workload.

    Mirrors a6_accreditation.build/main: idle_retained LP-active (central), grouped
    by cluster, /1e6 MW; keep clusters with mean eligible >= MIN_CLUSTER_MW.
    """
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    boundary = cfg.get("curtail_boundary", "idle_retained")
    elig = cfg.get("eligibility", "central")
    pod = pd.read_parquet(agg_path)
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

    curt = np.zeros((ncl, nt))
    work = np.zeros((ncl, nt))
    m = layer == "curtail"
    np.add.at(curt, (cl_codes[m], tpod[m]), pm.curtail_energy(parts, boundary)[m] / 1e6)
    np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
    good = work.sum(axis=0) > 0

    order = np.argsort(-curt.mean(axis=1))
    keep = [i for i in order if curt[i].mean() >= MIN_CLUSTER_MW]
    names = [str(cl_uniq[i])[:8] for i in keep]
    excluded = [str(cl_uniq[i])[:8] for i in order if i not in keep]

    C = curt[keep].astype(float)
    C[:, ~good] = np.nan
    W = work[keep].astype(float)
    W[:, ~good] = np.nan
    A = np.nansum(C, axis=0)
    A[~good] = np.nan
    Wf = np.nansum(W, axis=0)
    Wf[~good] = np.nan

    meta = dict(boundary=boundary, eligibility=elig, n_clusters=len(keep),
                cluster_ids=names, excluded_clusters=excluded,
                min_cluster_mw=MIN_CLUSTER_MW, valid_hours=int(good.sum()),
                calendar_hours=int(nt),
                mean_eligible_mw=float(np.nanmean(A)),
                mean_workload_mw=float(np.nanmean(Wf)),
                measured_scope_share=float(np.nanmean(A) / np.nanmean(Wf)),
                basis="GPU-side workload MW == facility-marginal at multiplier 1.0")
    return C, A, Wf, good, meta


# --------------------------------------------------------------------------- E1
def chain(C, A, mean_workload, T, alpha, s0, rng, n_shift):
    """P0..P3 and the three factors for one (T, alpha, s0).

    Primary P2 = independent (covariance-removed) aggregate firm power, so that
    a_coinc = P3/P2 = actual / independent < 1 matches the paper's indep_over_obs
    direction. P2_sum_of_quantiles is a labeled diagnostic only (the literal
    'sum of per-site quantiles' reading, which flips the coincidence sign)."""
    P1 = float(np.nanmean(A))                       # measured eligible mean level (flat)
    P0 = float(s0 * mean_workload)                  # straw man: assumed share * demand
    P2 = kval_decorrelated(C, T, alpha, rng, n_shift)   # independent-summed (decorrelated)
    P3 = float(kval(A, T, alpha))                   # honest coincident aggregate
    P2_sum = float(np.nansum([kval(C[i], T, alpha) for i in range(C.shape[0])]))
    return dict(P0=P0, P1=P1, P2=P2, P3=P3,
                a_scope=P1 / P0, a_persist=P2 / P1, a_coinc=P3 / P2,
                P2_sum_of_quantiles=P2_sum, a_coinc_sum_of_quantiles=P3 / P2_sum)


# --------------------------------------------------------------------------- E2
def planner_value_unscaled(content, C, A, meanA, T, alpha, rng, n_shift):
    """Value of the {dur?, coinc?} content, before the scope level scalar.
    dur toggles running-min quantile vs flat mean; coinc toggles the actual
    coincident aggregate vs the independent (decorrelated) aggregate."""
    dur = "dur" in content
    coinc = "coinc" in content
    if not dur:
        return meanA                                # flat: coincidence has no effect
    if coinc:
        return kval(A, T, alpha)                     # honest coincident aggregate
    return kval_decorrelated(C, T, alpha, rng, n_shift)   # independent (decorrelated)


def shapley(C, A, mean_workload, T, alpha, s0, rng, n_shift):
    meanA = float(np.nanmean(A))
    s_star = meanA / mean_workload
    scale_naive = s0 / s_star                       # = P0/P1
    contrib = {a: [] for a in ASSUMPTIONS}          # per-order marginal log-contributions
    per_order = {}
    unscaled_cache = {}                             # keyed by dur/coinc content only

    def logv(coal):
        # scope is a pure level scalar -> compute the unscaled content once and
        # apply the scope multiplier, so scope's marginal is exact (zero spread).
        content = frozenset(coal) - {"scope"}
        if content not in unscaled_cache:
            unscaled_cache[content] = planner_value_unscaled(content, C, A, meanA,
                                                             T, alpha, rng, n_shift)
        scale = 1.0 if "scope" in coal else scale_naive
        return np.log(scale * unscaled_cache[content])

    for order in itertools.permutations(ASSUMPTIONS):
        prefix = set()
        marg = {}
        for a in order:
            before = logv(prefix)
            prefix.add(a)
            marg[a] = logv(prefix) - before
            contrib[a].append(marg[a])
        per_order["->".join(order)] = {a: float(np.exp(marg[a])) for a in ASSUMPTIONS}
    shap_log = {a: float(np.mean(contrib[a])) for a in ASSUMPTIONS}
    spread = {a: {"factor_min": float(np.exp(min(contrib[a]))),
                  "factor_max": float(np.exp(max(contrib[a]))),
                  "log_spread": float(max(contrib[a]) - min(contrib[a]))}
              for a in ASSUMPTIONS}
    total_log = logv(set(ASSUMPTIONS)) - logv(set())
    assert abs(sum(shap_log.values()) - total_log) < 1e-9, "Shapley shares must sum to total"
    return dict(shapley_factor={a: float(np.exp(shap_log[a])) for a in ASSUMPTIONS},
                shapley_log=shap_log, spread=spread, per_order=per_order,
                total_log_overstatement=float(total_log))


# --------------------------------------------------------------------------- E3
def block_resample_indices(nt, block, rng):
    nblk = int(np.ceil(nt / block))
    starts = rng.integers(0, nt - block + 1, nblk)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:nt]
    return idx


def bootstrap(C, A, mean_workload, nboot, blocks, s0, n_shift, seed=20260915):
    nt = C.shape[1]
    out = {}
    for block in blocks:
        rng = np.random.default_rng(seed + block)
        samples = {(T, a): {k: [] for k in ("a_scope", "a_persist", "a_coinc",
                                            "shap_scope", "shap_dur", "shap_coinc")}
                   for T in HORIZONS for a in ALPHAS}
        for _ in range(nboot):
            idx = block_resample_indices(nt, block, rng)
            Cb = C[:, idx].copy()
            if block < nt:
                Cb[:, block - 1::block] = np.nan     # splice boundaries -> NaN gap
            Ab = np.nansum(Cb, axis=0)
            Ab[np.isnan(Cb).all(axis=0)] = np.nan
            for T in HORIZONS:
                for a in ALPHAS:
                    ch = chain(Cb, Ab, mean_workload, T, a, s0, rng, n_shift)
                    sh = shapley(Cb, Ab, mean_workload, T, a, s0, rng, n_shift)
                    s = samples[(T, a)]
                    s["a_scope"].append(ch["a_scope"])
                    s["a_persist"].append(ch["a_persist"])
                    s["a_coinc"].append(ch["a_coinc"])
                    s["shap_scope"].append(sh["shapley_factor"]["scope"])
                    s["shap_dur"].append(sh["shapley_factor"]["dur"])
                    s["shap_coinc"].append(sh["shapley_factor"]["coinc"])
        band = lambda v: {"p5": float(np.nanpercentile(v, 5)),
                          "p50": float(np.nanpercentile(v, 50)),
                          "p95": float(np.nanpercentile(v, 95))}
        out[str(block)] = {f"h{T}_a{a}": {k: band(v) for k, v in samples[(T, a)].items()}
                           for T in HORIZONS for a in ALPHAS}
    return out


# --------------------------------------------------------------------------- E4
def ensemble_structure(agg_path, cfg_path, n_ensemble, s0, n_shift, seed=7):
    """(a) Recompute the chain under the power-MC ensemble; does each factor stay
    <1 (a real derate) and monotone non-increasing in T across members?"""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    boundary = cfg.get("curtail_boundary", "idle_retained")
    elig = cfg.get("eligibility", "central")
    pod = pd.read_parquet(agg_path)
    pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
    nt = int(pod["t"].max()) + 1
    tpod = pod["t"].to_numpy()
    specs = sorted(pod["gpu_spec_public"].dropna().unique())
    spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
    cl_codes, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
    ncl = len(cl_uniq)
    layer = pm.assign_layer(pod, elig)
    m = layer == "curtail"
    rng = np.random.default_rng(seed)

    keep_ref = None
    rec = {f: {"a_persist": [], "a_coinc": []} for f in ("h1", "h4", "h24")}
    monotone_persist = monotone_coinc = all_lt1 = 0
    for b in range(n_ensemble):
        p = pm.draw_params(cfg, rng, specs, base=(b == 0))
        parts = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)
        curt = np.zeros((ncl, nt)); work = np.zeros((ncl, nt))
        np.add.at(curt, (cl_codes[m], tpod[m]), pm.curtail_energy(parts, boundary)[m] / 1e6)
        np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
        good = work.sum(axis=0) > 0
        if keep_ref is None:
            order = np.argsort(-curt.mean(axis=1))
            keep_ref = [i for i in order if curt[i].mean() >= MIN_CLUSTER_MW]
        C = curt[keep_ref].astype(float); C[:, ~good] = np.nan
        A = np.nansum(C, axis=0); A[~good] = np.nan
        mw = float(np.nanmean(np.where(good, work.sum(axis=0), np.nan)))
        persist = {}; coinc = {}
        for T in HORIZONS:
            ch = chain(C, A, mw, T, ALPHA_PRIMARY, s0, rng, n_shift)
            persist[T] = ch["a_persist"]; coinc[T] = ch["a_coinc"]
            rec[f"h{T}"]["a_persist"].append(ch["a_persist"])
            rec[f"h{T}"]["a_coinc"].append(ch["a_coinc"])
        mp = persist[1] >= persist[4] >= persist[24]
        mc = coinc[1] >= coinc[4] >= coinc[24]
        lt1 = all(persist[T] < 1 and coinc[T] < 1 for T in HORIZONS)
        monotone_persist += mp; monotone_coinc += mc; all_lt1 += lt1
    band = lambda v: {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                      "p95": float(np.percentile(v, 95))}
    return dict(n_ensemble=n_ensemble,
                fraction_persist_monotone_in_T=monotone_persist / n_ensemble,
                fraction_coinc_monotone_in_T=monotone_coinc / n_ensemble,
                fraction_all_factors_below_one=all_lt1 / n_ensemble,
                factor_bands={T: {"a_persist": band(rec[T]["a_persist"]),
                                  "a_coinc": band(rec[T]["a_coinc"])} for T in rec})


def external_coincidence(asi_pod, asi_server, helios_data, cfg_path, n_shift):
    """(b) a_coinc only, on external multi-site panels: structure replication."""
    import covariance_robustness_v2 as cov
    rng = np.random.default_rng(11)
    res = {}
    if asi_pod and asi_server:
        alloc, power, capacity, support = cov.load_asi(asi_pod, asi_server, cfg_path,
                                                       min_observed_fraction=0.8)
        C = power.to_numpy().T                      # per-cluster modeled workload MW
        A = np.nansum(C, axis=0)
        A[np.isnan(C).all(axis=0)] = np.nan
        res["ASI_long_support"] = _coinc_by_T(C, A, power.shape[1], rng, n_shift)
    if helios_data:
        alloc, cap = cov.load_helios(helios_data)
        C = alloc.to_numpy().T.astype(float)
        A = np.nansum(C, axis=0)
        res["Helios_allocated_GPU"] = _coinc_by_T(C, A, alloc.shape[1], rng, n_shift)
    return res


def _coinc_by_T(C, A, n_sites, rng, n_shift):
    out = {"n_sites": int(n_sites)}
    for a in (0.95,):
        vals = {}
        for T in HORIZONS:
            P2 = kval_decorrelated(C, T, a, rng, n_shift)
            P3 = float(kval(A, T, a))
            vals[f"h{T}"] = P3 / P2 if P2 else np.nan
        vals["coinc_below_one_all_T"] = bool(all(vals[f"h{T}"] < 1 for T in HORIZONS))
        vals["coinc_grows_with_T"] = bool(vals["h1"] >= vals["h4"] >= vals["h24"])
        out[f"a{a}"] = vals
    return out


# --------------------------------------------------------------------------- run
def run_all(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    C, A, Wf, good, meta = build_substrate(args.agg, args.config)
    mean_workload = meta["mean_workload_mw"]
    s_star = meta["measured_scope_share"]
    rng = np.random.default_rng(20260915)
    n_shift = args.n_shift

    # s0 sweep: measured scope, workload nominal flexible fraction, literature value.
    s0_sweep = {"measured_scope": s_star,
                "literature_value_PLACEHOLDER": args.s0_literature}
    if args.s0_workload_flexible is not None:
        s0_sweep["workload_nominal_flexible"] = args.s0_workload_flexible

    prov_common = dict(script_sha256=sha256(__file__),
                       powermodel_sha256=sha256(Path(__file__).parent / "powermodel.py"),
                       input_agg_sha256=sha256(args.agg),
                       config_sha256=sha256(args.config),
                       generated_at_utc=datetime.now(timezone.utc).isoformat(),
                       command=sys.argv)

    # E0 -----------------------------------------------------------------------
    e0 = dict(meta, quantity="P(x,T,alpha) = (1-alpha) quantile of T-hour running min",
              horizons=list(HORIZONS), alphas=list(ALPHAS), alpha_primary=ALPHA_PRIMARY,
              s0_sweep=s0_sweep, provenance=prov_common)
    (out / "E0_substrate.json").write_text(json.dumps(e0, indent=2))
    print(f"E0: N={meta['n_clusters']} clusters, {meta['valid_hours']} valid hours, "
          f"mean eligible {meta['mean_eligible_mw']:.3f} MW, scope share {s_star:.4f}", flush=True)

    # E1 -----------------------------------------------------------------------
    e1 = {"note": "Identity P3 = P0*a_scope*a_persist*a_coinc is exact by construction; "
                  "the contribution is the magnitude of each marginal factor, not the identity.",
          "by_s0": {}}
    for s0name, s0 in s0_sweep.items():
        e1["by_s0"][s0name] = {"s0": s0}
        for a in ALPHAS:
            for T in HORIZONS:
                ch = chain(C, A, mean_workload, T, a, s0, rng, n_shift)
                assert abs(ch["P3"] - ch["P0"] * ch["a_scope"] * ch["a_persist"]
                           * ch["a_coinc"]) < 1e-6 * max(ch["P3"], 1.0)
                ch["persistence_overstatement_pct"] = 100 * (1 / ch["a_persist"] - 1)
                ch["coincidence_overstatement_pct"] = 100 * (1 / ch["a_coinc"] - 1)
                e1["by_s0"][s0name][f"h{T}_a{a}"] = ch
    e1["provenance"] = prov_common
    (out / "E1_chain.json").write_text(json.dumps(e1, indent=2))
    for T in HORIZONS:
        ch = e1["by_s0"]["measured_scope"][f"h{T}_a{ALPHA_PRIMARY}"]
        print(f"E1 T={T}h a={ALPHA_PRIMARY}: persist over +{ch['persistence_overstatement_pct']:.1f}% "
              f"coinc over +{ch['coincidence_overstatement_pct']:.1f}%", flush=True)

    # E2 -----------------------------------------------------------------------
    e2 = {"by_s0": {}}
    for s0name, s0 in s0_sweep.items():
        e2["by_s0"][s0name] = {}
        for a in ALPHAS:
            for T in HORIZONS:
                e2["by_s0"][s0name][f"h{T}_a{a}"] = shapley(C, A, mean_workload, T, a, s0,
                                                            rng, n_shift)
    e2["provenance"] = prov_common
    (out / "E2_shapley.json").write_text(json.dumps(e2, indent=2))
    sp = e2["by_s0"]["measured_scope"][f"h4_a{ALPHA_PRIMARY}"]["spread"]
    print(f"E2 T=4h spread(log): scope {sp['scope']['log_spread']:.3e} "
          f"dur {sp['dur']['log_spread']:.3f} coinc {sp['coinc']['log_spread']:.3f}", flush=True)

    # E3 -----------------------------------------------------------------------
    if not args.skip_bootstrap:
        e3 = {"nboot": args.nboot, "blocks": list(BLOCKS),
              "note": "Preprocessing/masks fixed -> conditional intervals.",
              "n_shift": args.n_shift_boot,
              "measured_scope": bootstrap(C, A, mean_workload, args.nboot, BLOCKS,
                                          s_star, args.n_shift_boot)}
        e3["provenance"] = prov_common
        (out / "E3_bootstrap.json").write_text(json.dumps(e3, indent=2))
        print("E3: bootstrap done", flush=True)

    # E4 -----------------------------------------------------------------------
    e4 = {"analysis_choice_sensitivity": {
              "note": "alpha and s0 sweep are the full E1/E2 grids above.",
              "alphas": list(ALPHAS), "s0_sweep": list(s0_sweep)}}
    if not args.skip_ensemble:
        e4["ensemble_structure"] = ensemble_structure(args.agg, args.config,
                                                       args.n_ensemble, s_star,
                                                       args.n_shift_boot)
        es = e4["ensemble_structure"]
        print(f"E4a ensemble: persist monotone {es['fraction_persist_monotone_in_T']:.2f}, "
              f"coinc monotone {es['fraction_coinc_monotone_in_T']:.2f}, "
              f"all<1 {es['fraction_all_factors_below_one']:.2f}", flush=True)
    if args.asi_pod or args.helios_data:
        e4["external_coincidence"] = external_coincidence(
            args.asi_pod, args.asi_server, args.helios_data, args.config, n_shift)
        print("E4b external coincidence done", flush=True)
    e4["provenance"] = prov_common
    (out / "E4_robustness.json").write_text(json.dumps(e4, indent=2))
    print("ALL DONE", flush=True)


def self_test():
    """Identity + Shapley telescoping on a small synthetic 3-cluster substrate."""
    rng = np.random.default_rng(1)
    nt = 2000
    base = np.abs(rng.normal(2, 0.5, (3, nt)))          # per-cluster MW
    C = base.copy()
    A = C.sum(axis=0)
    mean_workload = 40.0
    for T in HORIZONS:
        for a in ALPHAS:
            ch = chain(C, A, mean_workload, T, a, 0.1, rng, 6)
            recon = ch["P0"] * ch["a_scope"] * ch["a_persist"] * ch["a_coinc"]
            assert abs(recon - ch["P3"]) < 1e-6 * max(ch["P3"], 1), (T, a, recon, ch["P3"])
            sh = shapley(C, A, mean_workload, T, a, 0.1, rng, 6)
            # scope has zero spread (pure level scalar)
            assert sh["spread"]["scope"]["log_spread"] < 1e-9
    bt = bootstrap(C, A, mean_workload, 20, (168,), 0.1, 3)
    assert "168" in bt and "h4_a0.95" in bt["168"]
    print("self-test passed: identity exact, Shapley telescopes, scope spread ~0, bootstrap runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", help="pod_hourly_agg.parquet")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "power_curves.yaml"))
    ap.add_argument("--out", help="output run directory")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--n-ensemble", type=int, default=200)
    ap.add_argument("--n-shift", type=int, default=12,
                    help="decorrelation shift draws for point-estimate independent planner")
    ap.add_argument("--n-shift-boot", type=int, default=3,
                    help="decorrelation shift draws inside bootstrap/ensemble (cost control)")
    ap.add_argument("--s0-literature", type=float, default=0.15,
                    help="PLACEHOLDER assumed interruptible share from planning "
                         "literature; author must set the real cited value")
    ap.add_argument("--s0-workload-flexible", type=float, default=None,
                    help="workload nominal flexible fraction (curtail+shift share)")
    ap.add_argument("--asi-pod")
    ap.add_argument("--asi-server")
    ap.add_argument("--helios-data")
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.agg and a.out):
        ap.error("--agg and --out are required unless --self-test")
    run_all(a)


if __name__ == "__main__":
    main()
