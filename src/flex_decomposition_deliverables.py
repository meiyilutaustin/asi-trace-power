#!/usr/bin/env python3
"""Manuscript deliverables from the E0-E4 decomposition JSONs: a three-stage
waterfall/funnel figure with bootstrap CIs, and a results table (Markdown)."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = "measured_scope"
HORIZONS = (1, 4, 24)


def load(run):
    run = Path(run)
    return {k: json.loads((run / f"{k}.json").read_text())
            for k in ("E0_substrate", "E1_chain", "E2_shapley",
                      "E3_bootstrap", "E4_robustness")}


def table_md(d, alpha=0.95):
    e1, e2, e3 = d["E1_chain"], d["E2_shapley"], d["E3_bootstrap"]
    b = e3[MS]["168"]
    out = ["# Decomposition results (13-cluster ASI internal, α=%.2f, idle-retained LP-active)\n" % alpha]
    e0 = d["E0_substrate"]
    out.append(f"Substrate: N={e0['n_clusters']} clusters, {e0['valid_hours']} valid hours, "
               f"mean eligible {e0['mean_eligible_mw']:.3f} MW, scope share "
               f"{e0['measured_scope_share']:.4f}. Firm power P(x,T,α) = (1−α) quantile "
               f"of the T-hour running minimum, GPU-side MW.\n")
    out.append("## E1 nested chain + E3 bootstrap CIs (block 168 h, 1000 rep)\n")
    out.append("| T (h) | P0=P1 (MW) | P2 (MW) | P3 (MW) | a_persist [p5,p95] | a_coinc [p5,p95] | persist over% | coinc over% |")
    out.append("|---|---|---|---|---|---|---|---|")
    for T in HORIZONS:
        c = e1["by_s0"][MS][f"h{T}_a{alpha}"]
        bp, bc = b[f"h{T}_a{alpha}"]["a_persist"], b[f"h{T}_a{alpha}"]["a_coinc"]
        out.append(f"| {T} | {c['P1']:.3f} | {c['P2']:.3f} | {c['P3']:.3f} | "
                   f"{c['a_persist']:.3f} [{bp['p5']:.3f}, {bp['p95']:.3f}] | "
                   f"{c['a_coinc']:.3f} [{bc['p5']:.3f}, {bc['p95']:.3f}] | "
                   f"+{c['persistence_overstatement_pct']:.1f}% | +{c['coincidence_overstatement_pct']:.1f}% |")
    out.append("\n## E2 Shapley (order-invariant) factors and across-order spread\n")
    out.append("| T (h) | scope | dur (persist) | coinc | log-spread dur | log-spread coinc |")
    out.append("|---|---|---|---|---|---|")
    for T in HORIZONS:
        s = e2["by_s0"][MS][f"h{T}_a{alpha}"]
        sf, sp = s["shapley_factor"], s["spread"]
        out.append(f"| {T} | {sf['scope']:.3f} | {sf['dur']:.3f} | {sf['coinc']:.3f} | "
                   f"{sp['dur']['log_spread']:.3f} | {sp['coinc']['log_spread']:.3f} |")
    out.append("\n## E4 robustness of the STRUCTURE\n")
    es = d["E4_robustness"].get("ensemble_structure", {})
    out.append(f"- (a) Power-MC ensemble (n={es.get('n_ensemble')}): all factors <1 in "
               f"{es.get('fraction_all_factors_below_one',0)*100:.0f}% of members; persistence "
               f"monotone in T in {es.get('fraction_persist_monotone_in_T',0)*100:.0f}%; coincidence "
               f"monotone in {es.get('fraction_coinc_monotone_in_T',0)*100:.0f}%.")
    ext = d["E4_robustness"].get("external_coincidence", {})
    for name, r in ext.items():
        v = r.get("a0.95", {})
        out.append(f"- (b) Second operator {name} (n_sites={r.get('n_sites')}): a_coinc = "
                   f"{v.get('h1'):.3f}/{v.get('h4'):.3f}/{v.get('h24'):.3f} at 1/4/24 h; "
                   f"<1 all T = {v.get('coinc_below_one_all_T')}, grows with T = {v.get('coinc_grows_with_T')}.")
    out.append("- (c) α∈{0.90,0.95,0.99} and s0 sweep: see E1_chain.json / E2_shapley.json by_s0 and by-α keys.")
    out.append("\n## Framework thesis\n")
    out.append("Duration is the common driver: the longer the event, the more BOTH the "
               "fixed-share (persistence) and site-independence (coincidence) assumptions "
               "overstate firm curtailable power, in the same direction.")
    return "\n".join(out) + "\n"


def waterfall(d, out_path, alpha=0.95, T=4):
    e1, e3 = d["E1_chain"], d["E3_bootstrap"]
    c = e1["by_s0"][MS][f"h{T}_a{alpha}"]
    b = e3[MS]["168"][f"h{T}_a{alpha}"]
    P1, P2, P3 = c["P1"], c["P2"], c["P3"]
    # stages: measured mean (P1) -> persistence -> coincidence (honest P3)
    labels = ["Measured\nmean eligible\n(P1)", "After\npersistence\n(P2)", "Honest firm\n(coincident, P3)"]
    vals = [P1, P2, P3]
    colors = ["#7f8c8d", "#2471A3", "#1E8449"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = np.arange(3)
    ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    # connectors + drop annotations
    for i in range(2):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [vals[i], vals[i]], color="#555", lw=1, ls=":", zorder=2)
    drop1 = (1 - c["a_persist"]) * 100
    drop2 = (1 - c["a_coinc"]) * 100
    ax.annotate(f"persistence\n−{drop1:.0f}% (×{c['a_persist']:.2f})", xy=(0.5, (P1 + P2) / 2),
                ha="center", va="center", fontsize=9, color="#2471A3")
    ax.annotate(f"coincidence\n−{drop2:.0f}% (×{c['a_coinc']:.2f})", xy=(1.5, (P2 + P3) / 2),
                ha="center", va="center", fontsize=9, color="#1E8449")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f} MW", ha="center", va="bottom", fontsize=9)
    # CI whisker on P3 via a_coinc*a_persist band (approximate, from factor CIs)
    lo = P1 * b["a_persist"]["p5"] * b["a_coinc"]["p5"]
    hi = P1 * b["a_persist"]["p95"] * b["a_coinc"]["p95"]
    ax.errorbar(2, P3, yerr=[[P3 - lo], [hi - P3]], fmt="none", ecolor="#154360", capsize=4, zorder=4)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Firm curtailable power (MW)")
    ax.set_ylim(0, P1 * 1.18)
    ax.set_title(f"Flexibility-overstatement decomposition ({T} h event, α={alpha})\n"
                 f"13-cluster ASI internal, idle-retained LP-active", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path + ".png", dpi=180)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d = load(a.run)
    (Path(a.run) / "RESULTS.md").write_text(table_md(d))
    waterfall(d, str(Path(a.run) / "waterfall_4h"), T=4)
    waterfall(d, str(Path(a.run) / "waterfall_24h"), T=24)
    print("wrote RESULTS.md, waterfall_4h.{png,pdf}, waterfall_24h.{png,pdf}")


if __name__ == "__main__":
    main()
