#!/usr/bin/env python3
"""Calendar-preserving Helios/ASI covariance comparison; no response eligibility.

Bootstrap resamples contiguous positions on the COMPLETE hourly calendar, jointly
across units. Masked hours have zero moment weight, never collapse time. Trends,
calendar profiles and masks are fixed from the original record (conditional CI).
ASI calendar is trace-relative. All four-cluster subsets are descriptive portfolio
comparisons, not independent replications. Allocated GPUs and modeled workload MW
are separate observables. No VC/workload-type scaling comparison is made.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import provenance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residuals(wide):
    raw = wide - wide.rolling(169, center=True, min_periods=24).mean()
    # Day-of-week offset only permutes the 168 groups; ASI dates remain unknown.
    key = np.arange(len(wide)) % 168
    return {'raw': raw, 'calendar': raw - raw.groupby(key).transform('mean')}


def masks(allocation, capacity):
    n = len(allocation)
    full = allocation.notna().all(axis=1).to_numpy()
    edge = np.ones(n, bool)
    edge[:168] = edge[-168:] = False
    baseline = capacity.reindex(index=allocation.index, columns=allocation.columns)
    low = ((allocation >= .25 * baseline) & baseline.gt(0)).all(axis=1).to_numpy()
    return {'full': full, 'trim': full & edge, 'low_allocation': full & low,
            'trim_low_allocation': full & edge & low}


def moments(x, valid):
    z = np.where(valid[:, None], x, 0.)
    return np.column_stack([valid.astype(float), z,
                            np.einsum('ti,tj->tij', z, z).reshape(len(x), -1)])


def covariance(moment):
    p = int(round(np.sqrt(moment.shape[-1] - .75) - .5))
    n = moment[..., 0]
    mu = moment[..., 1:1+p] / n[..., None]
    second = moment[..., 1+p:].reshape(*moment.shape[:-1], p, p) / n[..., None, None]
    return second - mu[..., :, None] * mu[..., None, :]


def block_draw_moments(m, block, nboot, seed):
    """Noncircular moving blocks; final block shortened to exactly n grid hours."""
    n = len(m)
    if n < block:
        raise ValueError('Calendar shorter than requested block')
    rng = np.random.default_rng(seed)
    cs = np.vstack([np.zeros((1, m.shape[1])), m.cumsum(axis=0)])
    lengths = [block] * (n // block) + ([n % block] if n % block else [])
    total = np.zeros((nboot, m.shape[1]))
    for length in lengths:
        starts = rng.integers(0, n - length + 1, nboot)
        total += cs[starts + length] - cs[starts]
    return total


def stats(cov):
    var = np.diagonal(cov, axis1=-2, axis2=-1).clip(0)
    vs = var.sum(axis=-1)
    numerator = cov.sum(axis=(-2, -1))
    denom = np.sqrt(var).sum(axis=-1)**2 - vs
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'R': np.sqrt(np.maximum(numerator / vs, 0)),
                'rho_weighted': (numerator - vs) / denom}


def beta_loo(wide, residual, mask):
    mu = wide.loc[mask].mean().to_numpy()
    sd = residual.loc[mask].std().to_numpy()
    valid = (mu > 0) & (sd > 0)
    x, y = np.log10(mu[valid]), np.log10(sd[valid])
    if len(x) < 3:
        return {'beta': None, 'leave_one_out': []}
    fit = lambda a, b: float(np.polyfit(a, b, 1)[0]) if np.ptp(a) > 0 else None
    return {'beta': fit(x, y), 'leave_one_out':
            [fit(np.delete(x, i), np.delete(y, i)) for i in range(len(x))],
            'n_clusters': len(x), 'interpretation': 'descriptive across clusters only'}


def analyse(name, wide, allocation, capacity, nboot, blocks):
    rows, descriptive = [], {}
    rr = residuals(wide)
    p = wide.shape[1]
    portfolios = [('all', tuple(range(p)))]
    if p > 4:
        portfolios += [('|'.join(str(wide.columns[i]) for i in ss), ss)
                       for ss in itertools.combinations(range(p), 4)]
    mm = masks(allocation, capacity)
    for maskname, mask in mm.items():
        descriptive[maskname] = beta_loo(wide, rr['raw'], mask)
        for kind, r in rr.items():
            x = r.to_numpy()
            valid = mask & np.isfinite(x).all(axis=1)
            m = moments(x, valid)
            point = covariance(m.sum(axis=0))
            for block in blocks:
                draws = block_draw_moments(m, block, nboot, 20260909 + block)
                good = draws[:, 0] >= 24
                cc = covariance(draws[good])
                for portfolio, idx in portfolios:
                    ix = np.array(idx)
                    obs = stats(point[np.ix_(ix, ix)])
                    boot = stats(cc[:, ix[:, None], ix])
                    row = dict(dataset=name, mask=maskname, residual=kind, portfolio=portfolio,
                               n_clusters=len(ix), hours=int(valid.sum()), calendar_hours=len(wide),
                               block_hours=block, valid_replicates=int(good.sum()))
                    for metric in obs:
                        lo, hi = np.nanpercentile(boot[metric], [5, 95])
                        row.update({metric: float(obs[metric]), metric+'_p5': lo, metric+'_p95': hi})
                    row['R_excludes_one'] = bool(row['R_p5'] > 1 or row['R_p95'] < 1)
                    row['rho_excludes_zero'] = bool(row['rho_weighted_p5'] > 0 or row['rho_weighted_p95'] < 0)
                    rows.append(row)
    return rows, descriptive


def load_helios(path):
    """Integrate running-job GPU occupancy over half-open hourly intervals."""
    start = pd.Timestamp('2020-04-01')
    idx = pd.date_range(start, '2020-09-28 23:00', freq='h')
    n = len(idx)
    data, caps = {}, {}
    for cluster in ['Earth', 'Saturn', 'Uranus', 'Venus']:
        f = Path(path)/cluster
        jobs = pd.read_csv(f/'cluster_log.csv', usecols=['start_time','end_time','gpu_num'])
        sh = ((pd.to_datetime(jobs.start_time)-start)/pd.Timedelta(hours=1)).to_numpy()
        eh = ((pd.to_datetime(jobs.end_time)-start)/pd.Timedelta(hours=1)).to_numpy()
        w = jobs.gpu_num.to_numpy(float)
        good = (eh > sh) & (w > 0) & (eh > 0) & (sh < n)
        sh, eh, w = np.clip(sh[good], 0, n), np.clip(eh[good], 0, n), w[good]
        # Piecewise-linear integrated occupancy; derivative jumps at start/end.
        edges = np.arange(n+1, dtype=float)
        integrals = []
        for times in [sh, eh]:
            order = np.argsort(times)
            tt, ww = times[order], w[order]
            k = np.searchsorted(tt, edges, side='left')
            integrals.append(edges*np.r_[0.,np.cumsum(ww)][k] - np.r_[0.,np.cumsum(ww*tt)][k])
        data[cluster] = np.diff(integrals[0]-integrals[1]).clip(0)
        cap = pd.read_csv(f/'cluster_gpu_number.csv', parse_dates=['date']).set_index('date')['total']
        cap = cap.reindex(idx.normalize()).ffill().bfill()
        caps[cluster] = cap.to_numpy()
    return pd.DataFrame(data, index=idx), pd.DataFrame(caps, index=idx)


def asi_valid_hours(allocation):
    good = allocation.sum(axis=1) > 0
    # Known invalid trace hour; retained as a gap even if aggregate rows exist.
    if 126 in good.index:
        good.loc[126] = False
    return good


def load_asi(path, server_path, config, min_observed_fraction=0.):
    import yaml
    import powermodel as pm
    pod = pd.read_parquet(path)
    pod['t'] = pod['day'].astype(int)*24 + pod['hour'].astype(int)
    nt = int(pod.t.max())+1
    allocation = pod.groupby(['t', 'cluster_id'], observed=True).gpu_hours.sum().unstack().reindex(range(nt))
    good = asi_valid_hours(allocation)
    allocation.loc[~good] = np.nan
    cfg = yaml.safe_load(Path(config).read_text())
    specs = sorted(pod.gpu_spec_public.dropna().unique())
    codes, uniq = pd.factorize(pod.gpu_spec_public)
    params = pm.draw_params(cfg, np.random.default_rng(2), specs, base=True)
    params['family'] = 'anchored'
    pod['mw'] = pm.row_energy(pod, cfg, params, codes, uniq)['total']/1e6
    power = pod.groupby(['t', 'cluster_id'], observed=True).mw.sum().unstack().reindex(range(nt))
    power.loc[~good] = np.nan
    # Same nontrivial power-support eligibility as historical A2; same units for both observables.
    minimum_hours = max(1500, int(np.ceil(min_observed_fraction * good.sum())))
    keep = power.columns[(power.mean() >= .01) & (power.notna().sum() >= minimum_hours)]
    srv = pd.read_parquet(server_path)
    srv["t"] = srv["day"].astype(int)*24 + srv["hour"].astype(int)
    capacity = srv.groupby(["t", "cluster_id"], observed=True).gpu_count.sum().unstack().reindex(range(nt))
    capacity.loc[~good] = np.nan
    support = pd.DataFrame({"mean_workload_MW": power.mean(),
                            "observed_hours": power.notna().sum(),
                            "selected": power.columns.isin(keep),
                            "minimum_observed_hours": minimum_hours})
    return allocation[keep], power[keep], capacity[keep], support


def self_test():
    fake = pd.DataFrame({"a": np.ones(200)})
    good_hour = asi_valid_hours(fake)
    assert not good_hour.loc[126] and good_hour.sum() == 199
    # Interior gap retains a calendar slot and zero moment weight.
    x = np.array([[1., 2.], [999., 999.], [3., 6.], [4., 8.]])
    valid = np.array([True, False, True, True])
    m = moments(x, valid)
    assert len(m) == 4 and not m[1].any()
    np.testing.assert_allclose(covariance(m.sum(0)), np.cov(x[valid], rowvar=False, ddof=0))
    cc = covariance(block_draw_moments(m, 2, 100, 3))
    cc = cc[np.diagonal(cc, axis1=-2, axis2=-1).sum(-1) > 0]
    np.testing.assert_allclose(stats(cc)['R'], 3/np.sqrt(5))
    # Identical clusters are perfectly weighted-correlated; R's null is one.
    np.testing.assert_allclose(stats(cc)['rho_weighted'], 1.)
    # Filling a gap then dropping it BEFORE bootstrap produces a different calendar length.
    assert len(m[valid]) != len(m)
    print('Calendar-gap, moment/covariance, perfect-correlation tests passed.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--helios-data')
    ap.add_argument('--asi-pod')
    ap.add_argument('--asi-server')
    ap.add_argument('--min-observed-fraction', type=float, default=0.,
                    help='ASI secondary long-coverage cohort, e.g.0.8; applied jointly with original1500h threshold')
    ap.add_argument('--config', default='configs/power_curves.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--blocks', type=int, nargs='+', default=[168, 336])
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    datasets = {}; hashes = {}
    if a.helios_data:
        alloc, cap = load_helios(a.helios_data)
        datasets['Helios_allocated_GPU'] = (alloc, alloc, cap)
        hashes.update({str(f): digest(f) for f in Path(a.helios_data).glob('*/*.csv')})
    if a.asi_pod:
        alloc, power, cap, support = load_asi(a.asi_pod, a.asi_server, a.config, a.min_observed_fraction)
        support.to_csv(out/"asi_cluster_support.csv")
        datasets['ASI_allocated_GPU'] = (alloc, alloc, cap)
        datasets['ASI_modeled_workload_MW'] = (power, alloc, cap)
        hashes[str(a.asi_pod)] = digest(a.asi_pod)
        hashes[str(a.asi_server)] = digest(a.asi_server)
        hashes[str(a.config)] = digest(a.config)
    if not datasets:
        raise ValueError('Provide at least one dataset')
    allrows = []; desc = {}
    for name, (wide, alloc, cap) in datasets.items():
        print(f'Analysing {name}: {wide.shape}', flush=True)
        rows, dd = analyse(name, wide, alloc, cap, a.nboot, a.blocks)
        allrows.extend(rows); desc[name] = dd
        pd.DataFrame(allrows).to_csv(out/'covariance_results.csv', index=False)
    table = pd.DataFrame(allrows)
    table[table.portfolio.eq('all')].to_csv(out/'full_portfolio.csv', index=False)
    subset = table[~table.portfolio.eq('all')]
    if len(subset):
        subset.groupby(['dataset','mask','residual','block_hours'])[['R','rho_weighted']].quantile([.05,.5,.95]).to_csv(out/'matched_four_summary.csv')
    (out/'descriptive_beta.json').write_text(json.dumps(desc, indent=2))
    prov = provenance.provenance()
    prov.update(arguments=vars(a), input_sha256=hashes, script_sha256=digest(__file__),
                fixed_preprocessing=True, missing_hours_preserved=True, ci_level=.90,
                asi_known_invalid_hour=126, asi_missing_cluster_hours="NaN, not zero-filled",
                supporting_code_sha256={f: digest(Path(__file__).parent/f) for f in ["powermodel.py", "provenance.py"]},
                mask_definition='allocation >= 25% published/inventory GPU capacity, all selected clusters; subsets use full-portfolio common support',
                primary='full', interpretation='historical covariance; no causal attribution or response eligibility')
    (out/'provenance.json').write_text(json.dumps(prov, indent=2))


if __name__ == '__main__':
    main()
