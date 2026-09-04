"""Shared util->power model for A1/A5/A6/A7b/A8 (revision 2, 2026-09-03).

Row model (per aggregate row = one (t, cluster, spec, job_type, priority,
state) cell with gpu_hours w, null-util gpu_hours wn and sufficient stats S_b):

    E_gpu = TDP * [ phi*w + (1-phi) * ( c0*w*[online] + g_eff(S) ) ]      (Wh)

  phi     idle fraction of TDP that an ALLOCATED GPU draws (A7: 0.17-0.21)
  c0      active-power floor for online inference expressed as a fraction of
          the idle->TDP span: c0 = (P_floor - phi)/(1-phi), P_floor ~ 0.43 TDP
          (decode is memory-bound: ML.ENERGY / Splitwise / 1/W-law; OSDI'26
          median SM util of GenAI serving = 6%).  The curve g(u) is then
          affinely compressed so saturation is unchanged:
              g_eff(u) = c0 + (1 - c0/g(1)) * g(u)
  g_eff   piecewise-linear in SM util on knots 0/.25/.5/.75/1, evaluated
          exactly from S_b = sum w*max(u-b,0); null-util rows use level ng.
  Standby rows draw phi*TDP*w only.

Curtailment boundaries for a preempted LP row (OSDI'26 §4.2: the container
exits and the GPU is RECLAIMED -> it keeps drawing phi*TDP as unallocated idle):
  attributed     phi*w + (1-phi)*active         (old headline; upper bound if
                                                 the whole pod power vanished)
  idle_retained  (1-phi)*active                 (default: GPU stays idle)
  node_sleep     attributed                     (node power-gated; = attributed
                                                 on the GPU side)
  +host          cpu_used_cores * (host_peak - host_idle) Wh (optional add-on)

Eligibility mappings (which running pods count as curtailable):
  conservative   LP and job_type in {offline_inference, training}
  central        all LP (spot: 60 s graceful preemption window, OSDI'26 §4.2)
  liberal        central + HP offline_inference (latency-insensitive per paper)
Layer precedence: standby > curtail (per mapping) > shift (HP training /
offline_inference not already curtail) > floor.
"""
import numpy as np

KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SHIFTABLE = ("offline_inference", "training")
BOUNDARIES = ("attributed", "idle_retained", "node_sleep")
ELIGIBILITY = ("conservative", "central", "liberal")


def curve_weights(kv):
    """c[j] with sum_rows w*g(u) = sum_j c[j]*S_{knot_j}."""
    slopes = np.diff(np.asarray(kv, float)) / np.diff(KNOTS)
    return np.concatenate([[slopes[0]], np.diff(slopes)])


def draw_params(cfg, rng, specs, base=False):
    """One parameter draw (base=True -> point estimates)."""
    p = {}
    u = (lambda k: cfg[k]["base"]) if base else \
        (lambda k: rng.uniform(*cfg[k]["range"]))
    p["idle_frac"] = u("idle_frac")
    p["null_g"] = u("null_util_g")
    p["pue"] = u("pue")
    of = cfg.get("online_floor_frac", {"base": None, "range": None})
    pf = None if of["base"] is None else (of["base"] if base else
                                          rng.uniform(*of["range"]))
    p["online_floor"] = pf
    p["c0"] = 0.0 if pf is None else max(0.0, (pf - p["idle_frac"])
                                         / (1 - p["idle_frac"]))
    h = cfg["host_w_per_core"]
    p["host_idle"] = h["idle"] if base else rng.uniform(*h["idle_range"])
    p["host_peak"] = h["peak"] if base else rng.uniform(*h["peak_range"])
    p["family"] = "anchored" if (base or rng.random() < 0.5) else "linear"
    tdp = dict(cfg["gpu_tdp_w"])
    ru = cfg["tdp_rel_uncertainty"]
    lo, hi = cfg["xpu_tdp_range_w"]
    p["tdp"] = {}
    for s in specs:
        if s in tdp:
            v = tdp[s]
            if isinstance(v, dict):          # explicit range, e.g. A100 PCIe/SXM
                p["tdp"][s] = v["base"] if base else rng.uniform(*v["range"])
            else:
                p["tdp"][s] = v if base else v * rng.uniform(1 - ru, 1 + ru)
        elif str(s).startswith("XPU"):
            p["tdp"][s] = 0.5 * (lo + hi) if base else rng.uniform(lo, hi)
        else:
            p["tdp"][s] = cfg["default_tdp_w"]
    return p


def tdp_of_rows(p, cfg, spec_codes, spec_uniq):
    return np.array([p["tdp"].get(s, cfg["default_tdp_w"])
                     for s in spec_uniq])[spec_codes]


def row_energy(pod, cfg, p, spec_codes, spec_uniq):
    """Per-row GPU energy parts (Wh): dict(idle, active, total)."""
    fam = cfg["curve_families"][p["family"]]
    tdp_row = tdp_of_rows(p, cfg, spec_codes, spec_uniq)
    S = pod[["S0", "S25", "S50", "S75"]].to_numpy()
    w = pod["gpu_hours"].to_numpy()
    wn = pod["gpu_hours_null_util"].to_numpy()
    jt = pod["job_type_public"].to_numpy()
    cw = np.zeros((len(pod), 4))
    named = [k for k in fam if k != "default"]
    for t_, kv in fam.items():
        mask = jt == t_ if t_ != "default" else ~np.isin(jt, named)
        cw[mask] = curve_weights(kv)
    active = (S * cw).sum(axis=1) + wn * p["null_g"]     # sum w*g(u)
    if p["c0"] > 0:
        online = jt == "online_inference"
        g1 = float(np.asarray(fam.get("online_inference",
                                      fam["default"]), float)[-1])
        scale = max(0.0, 1.0 - p["c0"] / g1)
        active[online] = p["c0"] * w[online] + scale * active[online]
    phi = p["idle_frac"]
    idle = tdp_row * phi * w
    act = tdp_row * (1 - phi) * active
    standby = (pod["state_public"].to_numpy() == "Standby")
    act[standby] = 0.0
    return {"idle": idle, "active": act, "total": idle + act}


def host_energy_rows(pod, p):
    """Dynamic host energy attributable to each row's used cores (Wh)."""
    if "cpu_used_cores" not in pod.columns:
        return None
    return pod["cpu_used_cores"].to_numpy() * (p["host_peak"] - p["host_idle"])


def assign_layer(pod, eligibility="central"):
    """Layer per row: standby / curtail / shift / floor (numpy str array)."""
    jt = pod["job_type_public"].astype(str).to_numpy()
    pr = pod["priority_class"].astype(str).to_numpy()
    standby = pod["state_public"].astype(str).to_numpy() == "Standby"
    lp = pr == "LP"
    hp = pr == "HP"
    shift_type = np.isin(jt, SHIFTABLE)
    if eligibility == "conservative":
        curtail = lp & shift_type
    elif eligibility == "central":
        curtail = lp
    elif eligibility == "liberal":
        curtail = lp | (hp & (jt == "offline_inference"))
    else:
        raise ValueError(eligibility)
    layer = np.full(len(pod), "floor", dtype=object)
    layer[hp & shift_type] = "shift"
    layer[curtail] = "curtail"
    layer[standby] = "standby"
    return layer


def curtail_energy(parts, boundary):
    if boundary == "attributed" or boundary == "node_sleep":
        return parts["total"]
    if boundary == "idle_retained":
        return parts["active"]
    raise ValueError(boundary)
