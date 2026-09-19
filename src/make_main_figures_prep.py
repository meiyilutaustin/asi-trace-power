#!/usr/bin/env python3
"""Compute all inputs for the 7 restructured manuscript figures (RESTRUCTURE.md).
Pure re-analysis on the exact E0-E4 substrate (no new fit / simulation).
Caches everything to figdata.pkl for the plotting script."""
import sys, os, json, pickle, time, tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np, pandas as pd, yaml
import flex_decomposition as fd
import powermodel as pm

AGG = "runs/2026-09-02_a1-load-reconstruction/pod_hourly_agg.parquet"
CFG = "configs/power_curves.yaml"
RUN = "runs/2026-09-15_flexibility-decomposition"
# Cache location: env-overridable, portable default (no dead session path).
OUT = os.environ.get("FIGCACHE",
                     os.path.join(tempfile.gettempdir(), "restructure_figdata.pkl"))

GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72]
ALPHAS = [0.90, 0.95, 0.99]
AP = 0.95
NSHIFT = 12
NMC = 200
NBOOT = 400
t_start = time.time()
def log(m): print("[%6.1fs] %s" % (time.time() - t_start, m), flush=True)

# ---- exact substrate --------------------------------------------------------
C, A, Wf, good, meta = fd.build_substrate(AGG, CFG)
nt = C.shape[1]
mw = meta["mean_workload_mw"]
# facility demand per hour (base point estimate) == paper's 52.32 MW facility
_fl = pd.read_parquet("runs/2026-09-04_rerun-p0/a1/fleet_hourly_power.parquet")
FAC = np.full(nt, np.nan)
_t = _fl["t"].to_numpy().astype(int)
_m = (_t >= 0) & (_t < nt)
FAC[_t[_m]] = _fl["base_mw"].to_numpy()[_m]
FAC = np.where(good, FAC, np.nan)
facility_mean = float(np.nanmean(FAC))
s0 = meta["measured_scope_share"]
meanA = float(np.nanmean(A))
log("substrate: %d clusters, meanA=%.4f MW, mw=%.4f, facility=%.2f, valid=%d"
    % (meta["n_clusters"], meanA, mw, facility_mean, meta["valid_hours"]))

def envT(series, alpha):
    return np.array([fd.kval(series, T, alpha) for T in GRID])

D = dict(meta=meta, GRID=GRID, ALPHAS=ALPHAS, AP=AP, meanA=meanA, mw=mw, s0=s0,
         facility_mean=facility_mean)

# =============================================================== FIG 2 =========
# (a) combined envelope, independent benchmark, 3 alphas, bootstrap band, powerMC
rng = np.random.default_rng(20260915)
D["f2_avg"] = meanA
D["f2_P3"] = {a: envT(A, a) for a in ALPHAS}                       # measured portfolio
D["f2_P2_indep"] = np.array([fd.kval_decorrelated(C, T, AP,
                    np.random.default_rng(7), NSHIFT) for T in GRID])
log("fig2 combined + independent done")

# (b) per-cluster normalized envelopes
D["f2_cluster_norm"] = np.array([envT(C[i], AP) / np.nanmean(C[i])
                                 for i in range(C.shape[0])])
D["f2_combined_norm"] = envT(A, AP) / meanA
log("fig2 per-cluster done")

# =============================================================== power-MC ======
# recompute C,A per draw; store P3@0.95 grid, factor overstatements, factors@1/4/24
cfg = yaml.safe_load(open(CFG))
pod = pd.read_parquet(AGG)
pod["t"] = pod["day"].astype(int) * 24 + pod["hour"].astype(int)
tpod = pod["t"].to_numpy()
specs = sorted(pod["gpu_spec_public"].dropna().unique())
spec_codes, spec_uniq = pd.factorize(pod["gpu_spec_public"])
cl_codes, cl_uniq = pd.factorize(pod["cluster_id"].astype(str))
ncl = len(cl_uniq)
layer_all = pm.assign_layer(pod, meta["eligibility"])
mask_curt = layer_all == "curtail"
mc_rng = np.random.default_rng(7)
keep_ref = None
mc_P3 = []          # 200 x len(GRID)  P3@0.95
mc_persOver = []    # 200 x len(GRID)
mc_coincOver = []   # 200 x len(GRID)
mc_share = []       # 200 x nt hourly LP share A/Wf
mc_fac = {T: {"persist": [], "coinc": []} for T in (1, 4, 24)}
for b in range(NMC):
    p = pm.draw_params(cfg, mc_rng, specs, base=(b == 0))
    parts = pm.row_energy(pod, cfg, p, spec_codes, spec_uniq)
    curt = np.zeros((ncl, nt)); work = np.zeros((ncl, nt))
    np.add.at(curt, (cl_codes[mask_curt], tpod[mask_curt]),
              pm.curtail_energy(parts, meta["boundary"])[mask_curt] / 1e6)
    np.add.at(work, (cl_codes, tpod), parts["total"] / 1e6)
    g = work.sum(axis=0) > 0
    if keep_ref is None:
        order = np.argsort(-curt.mean(axis=1))
        keep_ref = [i for i in order if curt[i].mean() >= fd.MIN_CLUSTER_MW]
    Cb = curt[keep_ref].astype(float); Cb[:, ~g] = np.nan
    Ab = np.nansum(Cb, axis=0); Ab[~g] = np.nan
    Wb = np.nansum(work, axis=0); Wb[~g] = np.nan
    p1 = float(np.nanmean(Ab))
    p3 = envT(Ab, AP)
    p2 = np.array([fd.kval_decorrelated(Cb, T, AP, np.random.default_rng(7), 3)
                   for T in GRID])
    mc_P3.append(p3)
    mc_persOver.append((p1 / p2 - 1) * 100)
    mc_coincOver.append((p2 / p3 - 1) * 100)
    mc_share.append(Ab / FAC)
    for T in (1, 4, 24):
        ch = fd.chain(Cb, Ab, float(np.nanmean(Wb)), T, AP, s0,
                      np.random.default_rng(7), 3)
        mc_fac[T]["persist"].append(ch["a_persist"])
        mc_fac[T]["coinc"].append(ch["a_coinc"])
    if b % 40 == 0:
        log("  power-MC draw %d/%d" % (b, NMC))
D["f2_mc_P3"] = np.array(mc_P3)
D["f6_mc_persOver"] = np.array(mc_persOver)
D["f6_mc_coincOver"] = np.array(mc_coincOver)
D["f3_mc_fac"] = {T: {k: np.array(v) for k, v in mc_fac[T].items()} for T in mc_fac}
log("power-MC ensemble (%d draws) done" % NMC)

# =============================================================== bootstrap =====
# block bootstrap of P3@0.95 grid and overstatements @ 3 alphas
def block_idx(nt, block, rng):
    return fd.block_resample_indices(nt, block, rng)
bt_rng = np.random.default_rng(20260915 + 168)
bt_P3 = []                                   # NBOOT x len(GRID)  @0.95
bt_over = {a: {"persist": [], "coinc": []} for a in ALPHAS}
bt_coincfac = []                             # a_coinc(T)@0.95
for _ in range(NBOOT):
    idx = block_idx(nt, 168, bt_rng)
    Cb = C[:, idx].copy(); Cb[:, 167::168] = np.nan
    Ab = np.nansum(Cb, axis=0); Ab[np.isnan(Cb).all(axis=0)] = np.nan
    p1 = float(np.nanmean(Ab))
    p3 = envT(Ab, AP)
    p2 = np.array([fd.kval_decorrelated(Cb, T, AP, np.random.default_rng(7), 3)
                   for T in GRID])
    bt_P3.append(p3)
    bt_coincfac.append(p3 / p2)
    for a in ALPHAS:
        p3a = envT(Ab, a)
        p2a = np.array([fd.kval_decorrelated(Cb, T, a, np.random.default_rng(7), 3)
                        for T in GRID])
        bt_over[a]["persist"].append((p1 / p2a - 1) * 100)
        bt_over[a]["coinc"].append((p2a / p3a - 1) * 100)
bt_P3 = np.array(bt_P3)
D["f2_boot_P3"] = {"p5": np.nanpercentile(bt_P3, 5, axis=0),
                   "p95": np.nanpercentile(bt_P3, 95, axis=0)}
D["f5_coinc_boot"] = {"p5": np.nanpercentile(np.array(bt_coincfac), 5, axis=0),
                      "p95": np.nanpercentile(np.array(bt_coincfac), 95, axis=0)}
D["f6_boot_over"] = {a: {k: {"p5": np.nanpercentile(np.array(v), 5, axis=0),
                             "p95": np.nanpercentile(np.array(v), 95, axis=0)}
                         for k, v in bt_over[a].items()} for a in ALPHAS}
log("bootstrap (%d reps) done" % NBOOT)

# =============================================================== FIG 3 =========
e1 = json.load(open(f"{RUN}/E1_chain.json"))["by_s0"]["measured_scope"]
e2 = json.load(open(f"{RUN}/E2_shapley.json"))["by_s0"]["measured_scope"]
e3 = json.load(open(f"{RUN}/E3_bootstrap.json"))["measured_scope"]["168"]
D["f3"] = {}
for T in (1, 4, 24):
    c = e1[f"h{T}_a0.95"]; s = e2[f"h{T}_a0.95"]; b = e3[f"h{T}_a0.95"]
    D["f3"][T] = dict(P1=c["P1"], P2=c["P2"], P3=c["P3"],
                      a_persist=c["a_persist"], a_coinc=c["a_coinc"],
                      boot_persist=(b["a_persist"]["p5"], b["a_persist"]["p95"]),
                      boot_coinc=(b["a_coinc"]["p5"], b["a_coinc"]["p95"]),
                      shap_dur_range=(s["spread"]["dur"]["factor_min"],
                                      s["spread"]["dur"]["factor_max"]),
                      shap_coinc_range=(s["spread"]["coinc"]["factor_min"],
                                        s["spread"]["coinc"]["factor_max"]))
log("fig3 from JSON done")

# =============================================================== FIG 4 =========
# (a) GPU-side hourly stacked composition (base params, point estimate)
p0 = pm.draw_params(cfg, np.random.default_rng(0), specs, base=True)
parts0 = pm.row_energy(pod, cfg, p0, spec_codes, spec_uniq)
host0 = pm.host_energy_rows(pod, p0)
jt = pod["job_type_public"].astype(str).to_numpy()
pri = pod["priority_class"].astype(str).to_numpy()
idle = parts0["idle"] / 1e6
act = parts0["active"] / 1e6
lay = layer_all
def acc(vals, mask):
    out = np.zeros(nt); np.add.at(out, tpod[mask], vals[mask]); return out
comp = {}
comp["gpu_idle"] = acc(idle, np.ones(len(pod), bool))
comp["lp_active"] = acc(act, lay == "curtail")            # highlight
comp["hp_shift_active"] = acc(act, lay == "shift")
comp["floor_active"] = acc(act, (lay != "curtail") & (lay != "shift"))
if host0 is not None:
    comp["host"] = acc(host0 / 1e6, np.ones(len(pod), bool))
it_total = comp["gpu_idle"] + comp["lp_active"] + comp["hp_shift_active"] + \
           comp["floor_active"] + comp.get("host", 0.0)
comp["facility_overhead"] = (p0["pue"] - 1.0) * it_total
modelled = it_total + comp["facility_overhead"]
comp["other_nongpu"] = np.maximum(FAC - modelled, 0.0)
for k in comp:
    comp[k] = np.where(good, comp[k], np.nan)
D["f4_comp"] = comp
D["f4_good"] = good
# (b) hourly LP share ECDF + powerMC band
share = np.where(good, A / FAC, np.nan)
sv = np.sort(share[~np.isnan(share)])
probs = np.linspace(0, 1, 200)
D["f4_share_ecdf_x"] = np.quantile(sv, probs)
D["f4_share_ecdf_p"] = probs
mc_share = np.array(mc_share)
mc_q = np.array([np.nanquantile(row, probs) for row in mc_share])   # NMC x 200
D["f4_share_band"] = {"p5": np.nanpercentile(mc_q, 5, axis=0),
                      "p95": np.nanpercentile(mc_q, 95, axis=0)}
D["f4_share_mean"] = float(np.nanmean(share))
# (c) scope staircase: LP -> +offline_inference -> +HP training (mean & 4h/0.95)
def elig_series(mask_extra):
    curt = np.zeros((ncl, nt))
    mm = mask_curt | mask_extra
    np.add.at(curt, (cl_codes[mm], tpod[mm]),
              pm.curtail_energy(parts0, meta["boundary"])[mm] / 1e6)
    Cs = curt[keep_ref].astype(float); Cs[:, ~good] = np.nan
    As = np.nansum(Cs, axis=0); As[~good] = np.nan
    return float(np.nanmean(As)), float(fd.kval(As, 4, AP))
steps = {}
steps["LP"] = elig_series(np.zeros(len(pod), bool))
steps["+offline inf"] = elig_series((jt == "offline_inference") & (pri != "LP"))
steps["+HP train"] = elig_series(((jt == "offline_inference") | (jt == "training"))
                                 & (pri == "HP"))
D["f4_steps"] = steps
log("fig4 done")

# =============================================================== FIG 5 =========
# (a) heatmap: per-cluster daily-mean eligible normalized to own mean + combined
ndays = nt // 24
def daily(series):
    s = series[:ndays * 24].reshape(ndays, 24)
    return np.nanmean(s, axis=1)
Cd = np.array([daily(C[i]) / np.nanmean(C[i]) for i in range(C.shape[0])])
Ad = daily(A) / meanA
D["f5_heat_clusters"] = Cd
D["f5_heat_combined"] = Ad
# below-own-10th-percentile shortfall indicator (spec Fig 5a): fraction of each
# day's ACTIVE hours below that cluster's own 10th pct; NaN where inactive.
def shortfall(series):
    act = series > 0
    if act.sum() == 0:
        return np.full(ndays, np.nan)
    p10 = np.quantile(series[act], 0.10)
    ind = (act & (series < p10)).astype(float)
    ind[~act] = np.nan
    d = ind[:ndays * 24].reshape(ndays, 24)
    frac = np.nanmean(d, axis=1)          # NaN where whole day inactive
    return frac
D["f5_short_clusters"] = np.array([shortfall(C[i]) for i in range(C.shape[0])])
D["f5_short_combined"] = shortfall(A)
# order clusters by mean level (descending) for readability
D["f5_heat_order"] = np.argsort(-np.array([np.nanmean(C[i]) for i in range(C.shape[0])]))
# (b) a_coinc(T)@0.95 measured + independence null band
D["f5_coinc_meas"] = D["f2_P3"][AP] / D["f2_P2_indep"]
null = []
nr = np.random.default_rng(101)
for _ in range(200):
    tr = fd.trend_of(C); res = C - tr
    s = np.zeros(nt)
    for i in range(C.shape[0]):
        s += np.roll(res[i], nr.integers(0, nt)) + tr[i]
    s = np.maximum(s, 0.0)
    p3n = envT(s, AP)
    p2n = np.array([fd.kval_decorrelated(C, T, AP, np.random.default_rng(7), 3)
                    for T in GRID])
    null.append(p3n / p2n)
null = np.array(null)
D["f5_null"] = {"p5": np.nanpercentile(null, 5, axis=0),
                "p50": np.nanpercentile(null, 50, axis=0),
                "p95": np.nanpercentile(null, 95, axis=0)}
# (c) Helios at 1/4/24
e4 = json.load(open(f"{RUN}/E4_robustness.json"))
hel = e4["external_coincidence"]["Helios_allocated_GPU"]["a0.95"]
D["f5_helios"] = {1: hel["h1"], 4: hel["h4"], 24: hel["h24"]}
log("fig5 done")

# =============================================================== FIG 6 =========
D["f6_persOver"] = {a: (D["f2_avg"] / np.array([fd.kval_decorrelated(
        C, T, a, np.random.default_rng(7), NSHIFT) for T in GRID]) - 1) * 100
    for a in ALPHAS}
D["f6_coincOver"] = {}
for a in ALPHAS:
    p2 = np.array([fd.kval_decorrelated(C, T, a, np.random.default_rng(7), NSHIFT)
                   for T in GRID])
    p3 = envT(A, a)
    D["f6_coincOver"][a] = (p2 / p3 - 1) * 100
D["f6_helios_coincOver"] = {T: (1.0 / D["f5_helios"][T] - 1) * 100 for T in (1, 4, 24)}
# (b) Shapley log-spread(T) dur & coinc
dur_sp, coinc_sp = [], []
for T in GRID:
    sh = fd.shapley(C, A, mw, T, AP, s0, np.random.default_rng(7), 6)
    dur_sp.append(sh["spread"]["dur"]["log_spread"])
    coinc_sp.append(sh["spread"]["coinc"]["log_spread"])
D["f6_shap_dur"] = np.array(dur_sp)
D["f6_shap_coinc"] = np.array(coinc_sp)
log("fig6 done")

# =============================================================== FIG 7 =========
COV = np.linspace(0.80, 0.999, 40)
# (b) grid company: P3(T,alpha) vs coverage, T=1/4/24
D["f7_cov"] = COV
D["f7_grid"] = {T: np.array([fd.kval(A, T, a) for a in COV]) for T in (1, 4, 24)}
# (a) planner: additional connectable load (MW) from the peak-cap interconnection
# screen against the EIA-930 regional demand series (A8), NOT the raw envelope.
# Bugfix 2026-09-17: the earlier version plotted the 4 h envelope (single-digit MW)
# under a "connectable load" axis; the correct quantity is the A8 headroom screen,
# tier facility_marginal_pue1.0 (matches the paper's 1 MW-suppressed = 1 MW-facility
# marginal assumption), cap margin m0.05 (the values quoted in the Discussion:
# no-flex 1684, measured 1809, fixed-20% 2105 MW).
_A8 = json.load(open("runs/2026-09-04_prewrite/a8_v2b/summary.json"))
_a8t = _A8["tiers"]["facility_marginal_pue1.0"]
_A8_MARGIN = "m0.05"
_bud_pct = _A8["settings"]["budgets_pct_hours"]          # [0.25, 0.5, 1.0, 2.0]
D["f7a_budget_h"] = np.array(_bud_pct) / 100.0 * 8760.0  # hours per year
_hm = _a8t["headroom_mw"][_A8_MARGIN]
# budget keys are '0.25%','0.5%','1.0%','2.0%'; match with a graceful fallback
def _seq(scen, stat):
    d = _hm[scen]
    out = []
    for p in _bud_pct:
        k = f"{p:g}%"
        if k not in d:
            k = f"{p}%"
        out.append(d[k][stat])
    return np.array(out)
D["f7a_fixed"]        = _seq("const20", "p50")           # 20% fixed-share ~2105 (flat)
D["f7a_measured"]     = _seq("measured", "p50")          # measured envelope ~1809 (flat)
D["f7a_measured_p5"]  = _seq("measured", "p5")
D["f7a_measured_p95"] = _seq("measured", "p95")
D["f7a_full"]         = _seq("full", "p50")              # fully curtailable, rises w/ budget
_nf = _a8t["headroom_no_flex_mw"][_A8_MARGIN]["p50"]     # scalar ~1684
D["f7a_noflex"]       = np.full(len(_bud_pct), _nf)
# (c) operator: P3(4h,alpha) vs HP participation f in [0,1]
Chp = np.zeros((ncl, nt))
mhp = (pri == "HP") & np.isin(jt, ("offline_inference", "training"))
np.add.at(Chp, (cl_codes[mhp], tpod[mhp]),
          pm.curtail_energy(parts0, meta["boundary"])[mhp] / 1e6)
Chp = Chp[keep_ref].astype(float); Chp[:, ~good] = np.nan
FR = np.linspace(0, 1, 25)
D["f7_frac"] = FR
D["f7_oper"] = {}
for a in ALPHAS:
    ys = []
    for f in FR:
        Ef = np.nansum(C + f * Chp, axis=0); Ef[~good] = np.nan
        ys.append(fd.kval(Ef, 4, a))
    D["f7_oper"][a] = np.array(ys)
log("fig7 done")

pickle.dump(D, open(OUT, "wb"))
log("WROTE %s" % OUT)
print("SANITY P3@0.95:", dict(zip(GRID, np.round(D["f2_P3"][0.95], 3))))
print("SANITY persOver@0.95:", dict(zip(GRID, np.round(D["f6_persOver"][0.95], 1))))
print("SANITY coincOver@0.95:", dict(zip(GRID, np.round(D["f6_coincOver"][0.95], 1))))
