"""Common-bin, CPU-utilization-based M100 temporal holdout validation.
No job labels or demand-response interpretation. Raw/cache outputs outside git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from m100_validation import IPMI_ALL, IPMI_HOST, member_path, load_logics, check_c_pue, load_llnl
from provenance import provenance

RUN = '2026-09-09_m100-validation-v2'
MONTHS = ['22-03', '22-07', '22-09']

def q(s):
    return "'" + str(s).replace("'", "''") + "'"

def metrics():
    return [('ipmi_pub', m) for m in IPMI_ALL] + [('ganglia_pub', m) for m in
        [f'Gpu{i}_{v}' for i in range(4) for v in ['gpu_utilization', 'power_usage']]
        + ['cpu_idle', 'cpu_num']]

def aggregate(con, data, month, plugin, metric, cache, audit):
    p = Path(member_path(str(data), month, plugin, metric))
    if not p.exists():
        raise FileNotFoundError(f'Required measured feature missing: {p}; no allocation substitute allowed')
    name = f'{plugin}__{metric}'
    dest = cache / f'{name}.parquet'
    if not dest.exists():
        # Identical timestamp duplicates collapse before binning. Contradictory
        # duplicates are averaged and counted by the separate audit query.
        sql = f"""SELECT try_cast(node AS INTEGER) node,
          floor(epoch(timestamp)/300)::BIGINT bin, avg(value)::DOUBLE AS "value",
          count(*)::INTEGER n, avg(CASE WHEN value>=0 AND value<=100 THEN
          60+240*CASE WHEN value<=25 THEN value/50.0
          WHEN value<=50 THEN .5+(value-25)/100.0
          WHEN value<=75 THEN .75+(value-50)*.006
          ELSE .9+(value-75)*.004 END END) raw_curve
          FROM (SELECT node,timestamp,avg(try_cast(value AS DOUBLE)) AS "value"
          FROM read_parquet({q(p)},hive_partitioning=false)
          WHERE try_cast(node AS INTEGER) BETWEEN 0 AND 979
          GROUP BY node,timestamp) GROUP BY node,bin"""
        con.execute(f'COPY ({sql}) TO {q(dest)} (FORMAT PARQUET)')
    # Count and range audit from compact bins, never silently clip utilization.
    a = con.execute(f"SELECT count(*),sum(n),min(value),max(value) FROM read_parquet({q(dest)})").fetchone()
    audit[name] = dict(zip(['bins','distinct_timestamps','min','max'], a))
    count=con.execute(f'SELECT count(*) FROM read_parquet({q(p)},hive_partitioning=false) WHERE try_cast(node AS INTEGER) BETWEEN 0 AND 979').fetchone()[0]
    audit[name]['raw_compute_rows']=count
    audit[name]['duplicate_timestamp_rows']=count-int(a[1])
    return dest

def panel(data, cache, month, threads):
    d = cache/month; d.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(d/'aggregate.duckdb'))
    con.execute(f"SET threads={int(threads)}")
    con.execute("SET memory_limit='12GB'")
    con.execute(f"SET temp_directory={q(d/'spill')}")
    audit={}; paths=[]
    for pl, met in metrics():
        print(month, met, flush=True)
        paths.append((met, aggregate(con,data,month,pl,met,d,audit)))
    selects=['t0.node','t0.bin']; joins=[]; conditions=[]
    for i,(met,p) in enumerate(paths):
        alias=f't{i}'
        selects += [f'{alias}.value AS "{met}"']
        if met.endswith('gpu_utilization'):
            selects += [f'{alias}.raw_curve AS "{met}_raw"']
            conditions += [f'{alias}.value BETWEEN 0 AND 100']
        elif met=='cpu_idle': conditions += [f'{alias}.value BETWEEN 0 AND 100']
        elif met=='cpu_num': conditions += [f'{alias}.value > 0']
        elif met in IPMI_HOST: conditions += [f'{alias}.value >= 0']
        else: conditions += [f'{alias}.value > 0']
        conditions += [f'{alias}.n >= {2 if met.startswith("cpu_") else 12}']
        joins += [f'{"FROM" if i==0 else "INNER JOIN"} read_parquet({q(p)}) {alias}' +
                  ('' if i==0 else f' ON t0.node={alias}.node AND t0.bin={alias}.bin')]
    # Retain common five-minute support and report its exact fraction per hour.
    common='SELECT '+','.join(selects)+' '+' '.join(joins)+' WHERE '+' AND '.join(conditions)
    con.execute(f'CREATE OR REPLACE TABLE common AS {common}')
    audit['common_bins']=con.execute('SELECT count(*) FROM common').fetchone()[0]
    cols=[f'avg("{m}") AS "{m}"' for m,_ in paths]
    cols += [f'avg("Gpu{i}_gpu_utilization_raw") AS "gpu{i}_raw"' for i in range(4)]
    h=con.execute('SELECT node,floor(bin/12)::BIGINT AS "hour",count(*) bins,'+','.join(cols)+
                  ' FROM common GROUP BY node,hour').fetchdf()
    for i in range(4):
        for sensor,short in [('power_management_limit','cap'),('sm_clock','clock')]:
            met=f'Gpu{i}_{sensor}'
            if Path(member_path(str(data),month,'ganglia_pub',met)).exists():
                extra=aggregate(con,data,month,'ganglia_pub',met,d,audit)
                e=con.execute(f'SELECT node,floor(bin/12)::BIGINT AS "hour",avg(value) gpu{i}_{short} FROM read_parquet({q(extra)}) WHERE n>=12 GROUP BY node,hour HAVING count(*)>=11').fetchdf()
                h=h.merge(e,on=['node','hour'],how='left',validate='one_to_one')
            else: audit[met]={'status':'absent: no cap/clock inference'}
    h.to_parquet(d/'node_hour.parquet',index=False)
    (d/'coverage.json').write_text(json.dumps(audit,indent=2))
    con.close()
    return h,audit

def curve(u):
    return np.interp(np.asarray(u)/100,[0,.25,.5,.75,1],[60,180,240,276,300])

def summarize(pred, obs):
    pred=np.asarray(pred); obs=np.asarray(obs); good=np.isfinite(pred)&np.isfinite(obs)&(obs>0)
    pred=pred[good];obs=obs[good]
    if not len(obs): return {'n':0}
    return {'n':len(obs),'energy_bias':float(pred.sum()/obs.sum()-1),
            'mae_w':float(np.mean(abs(pred-obs))),
            'relative_error_quantiles':np.quantile(pred/obs-1,[.01,.05,.5,.95,.99]).tolist(),
            'observed_mean_w':float(obs.mean()),'predicted_mean_w':float(pred.mean())}

def fit(march):
    # Deterministic rack holdout: 39 calibration racks, 10 held-out racks.
    train=march[(march.bins>=11)&((march.node//20)%5!=0)]
    u=np.concatenate([train[f'Gpu{i}_gpu_utilization'].values/100 for i in range(4)])
    p=np.concatenate([train[f'Gpu{i}_power_usage'].values for i in range(4)])
    # Nonnegative hinge increments guarantee a monotone, bounded-by-fit curve.
    B=np.column_stack([np.ones(len(u))]+[np.clip((u-k)/.25,0,1) for k in [0,.25,.5,.75]])
    weights,_=nnls(B,p)
    x=32*(1-train.cpu_idle.values/100)
    host=train[IPMI_HOST].sum(axis=1).values
    host_weights,_=nnls(np.column_stack([np.ones(len(x)),x]),host)
    gpu=train[[f'Gpu{i}_power_usage' for i in range(4)]].sum(axis=1).values
    psu=train.ps0_input_power.values+train.ps1_input_power.values
    # Independently meter-derived AC remainder; not fitted to prediction errors.
    remainder=float(np.mean(psu-gpu-host))
    # Rack-resampled parameter uncertainty is separate from holdout residuals.
    racks=(train.node.values//20);unique=np.unique(racks);gpu_racks=np.tile(racks,4)
    H=np.column_stack([np.ones(len(x)),x]); moments=[]
    for rack in unique:
        br=B[gpu_racks==rack];pr=p[gpu_racks==rack]
        hr=H[racks==rack];yr=host[racks==rack]
        moments.append((br.T@br,br.T@pr,hr.T@hr,hr.T@yr))
    rng=np.random.default_rng(20260909); draws=[]
    for rep in range(100):
        pick=rng.integers(0,len(unique),len(unique));total=[sum(moments[i][j] for i in pick) for j in range(4)]
        gw=[]
        for gram,rhs in [(total[0],total[1]),(total[2],total[3])]:
            L=np.linalg.cholesky(gram+np.eye(len(rhs))*1e-8)
            coef,_=nnls(L.T,np.linalg.solve(L,rhs));gw.append(coef)
        draws.append(np.r_[gw[0][0],gw[0][0]+np.cumsum(gw[0][1:]),gw[1]])
    parameter_intervals=np.quantile(draws,[.025,.5,.975],axis=0).tolist()
    return {'training_total_tdp_knots':(np.r_[weights[0],weights[0]+np.cumsum(weights[1:])]/300).tolist(),
            'gpu_increment_w':weights.tolist(),'host_intercept_slope':host_weights.tolist(),
            'metered_ac_remainder_w':remainder,'calibration_node_hours':len(train),
            'calibration':'March, node//20 % 5 != 0; bins>=11',
            'parameter_rack_bootstrap_100_q025_q50_q975':parameter_intervals,
            'parameter_interval_columns':['gpu_W_u0','gpu_W_u25','gpu_W_u50','gpu_W_u75','gpu_W_u100','host_intercept_W','host_slope_W_per_active_core'],
            'host_feature':'32 physical cores * (1 - measured cpu_idle/100)',
            'boundary':'PSU input; separately measured non-GPU/non-host remainder held fixed'}

def evaluate(h,model,out):
    out.mkdir(parents=True,exist_ok=True)
    h=h.copy(); result={}
    u=h[[f'Gpu{i}_gpu_utilization' for i in range(4)]].values/100
    gpu0=np.sum(curve(u*100),axis=1)
    w=np.array(model['gpu_increment_w'])
    gpu1=np.sum(w[0]+sum(w[j+1]*np.clip((u-k)/.25,0,1) for j,k in enumerate([0,.25,.5,.75])),axis=1)
    host0=48+80*(1-h.cpu_idle.values/100)
    a,b=model['host_intercept_slope'];host1=a+b*32*(1-h.cpu_idle.values/100)
    obs_gpu=h[[f'Gpu{i}_power_usage' for i in range(4)]].sum(axis=1).values
    obs_host=h[IPMI_HOST].sum(axis=1).values
    h['observed_psu']=h.ps0_input_power+h.ps1_input_power
    remainder=model['metered_ac_remainder_w']
    for tag,gpu,host in [('original',gpu0,host0),('gpu_only',gpu1,host0),('host_only',gpu0,host1),('both',gpu1,host1)]:
        h[tag]=gpu+host+remainder
        h[tag+'_no_boundary_correction']=gpu+host
        result[tag]={}
        for panel_name,mask in [('primary',h.bins>=11),('lower_coverage',h.bins>=8),('rack_holdout',(h.bins>=11)&((h.node//20)%5==0))]:
            result[tag][panel_name]={'gpu':summarize(gpu[mask],obs_gpu[mask]),'host':summarize(host[mask],obs_host[mask]),
                'node_psu':summarize(h.loc[mask,tag],h.loc[mask,'observed_psu']),
                'node_psu_no_boundary_correction':summarize(h.loc[mask,tag+'_no_boundary_correction'],h.loc[mask,'observed_psu'])}
    h=h[h.bins>=11]
    raw=h[[f'gpu{i}_raw' for i in range(4)]].sum(axis=1)
    hm=sum(curve(h[f'Gpu{i}_gpu_utilization']) for i in range(4))
    result['jensen']={'quantiles':np.quantile(hm/raw-1,[.5,.95,.99]).tolist(), 'energy_weighted':float(hm.sum()/raw.sum()-1)}
    caps=[f'gpu{i}_cap' for i in range(4)]
    if all(c in h for c in caps):
        complete_cap=h[caps].notna().all(axis=1)
        result['power_cap_strata']={}
        for name,mask in [('any_below_299W',complete_cap&(h[caps].min(axis=1)<299)),('all_at_least_299W',complete_cap&(h[caps].min(axis=1)>=299))]:
            result['power_cap_strata'][name]={tag:summarize(h.loc[mask,tag],h.loc[mask,'observed_psu']) for tag in ['original','gpu_only','host_only','both']}
    clocks=[f'gpu{i}_clock' for i in range(4)]
    if all(c in h for c in clocks):
        c=h[clocks].mean(axis=1).where(h[clocks].notna().all(axis=1))
        result['sm_clock_mhz_quantiles']=c.quantile([.05,.5,.95]).to_dict()
    result['cpu_num_quantiles']=h.cpu_num.quantile([0,.5,1]).to_dict()
    cols=['observed_psu','original','gpu_only','host_only','both']
    cluster=h.groupby('hour')[cols].sum();cluster['n_nodes']=h.groupby('hour').size()
    # Partial sums retained with coverage, never labeled full switchboard accuracy.
    cluster.to_csv(out/'cluster_common_support.csv')
    result['temporal_uncertainty']={}
    if len(cluster):
        calendar=cluster.reindex(range(int(cluster.index.min()),int(cluster.index.max())+1))
        rng=np.random.default_rng(20260909)
        for block in [168,336]:
            estimates={tag:[] for tag in ['original','gpu_only','host_only','both']}
            n=len(calendar)
            for rep in range(200):
                idx=np.concatenate([(int(start)+np.arange(min(block,n)))%n for start in rng.integers(0,n,int(np.ceil(n/min(block,n))))])[:n]
                sample=calendar.iloc[idx]
                denom=sample.observed_psu.sum()
                for tag in estimates:estimates[tag].append(sample[tag].sum()/denom-1)
            result['temporal_uncertainty'][str(block)]={tag:np.quantile(vals,[.025,.5,.975]).tolist() for tag,vals in estimates.items()}
        result['temporal_uncertainty_note']='200 circular calendar-block resamples; preprocessing and March parameters fixed; paired observed-support energy bias, not parameter CI or full-fleet extrapolation'

    complete=cluster[cluster.n_nodes==980].copy()
    result['complete_fleet_hours']=len(complete)
    result['max_reporting_nodes']=int(cluster.n_nodes.max()) if len(cluster) else 0
    shape=[]
    for col in cols:
        s=complete[col]; full=s.reindex(range(int(s.index.min()),int(s.index.max())+1)) if len(s) else s
        row={'model':col,'hours':len(s),'load_factor':s.mean()/s.max() if len(s) else None}
        for duration in [1,4,24]:
            r=full.rolling(duration,min_periods=duration).min().dropna()
            for quant in [.05,.5,.95]:row[f'min_{duration}h_q{quant}']=r.quantile(quant) if len(r) else None
        shape.append(row)
    pd.DataFrame(shape).to_csv(out/'complete_fleet_shape.csv',index=False)
    if len(complete):
        clock=complete[cols].groupby(complete.index%24).mean()/complete[cols].mean()
        clock.to_csv(out/'complete_fleet_clock_normalized.csv')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--llnl',type=Path);ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--months',nargs='+',default=MONTHS,help='panel/fit/evaluate months; must include 22-03 (calibration)')
    ap.add_argument('--facility-months',nargs='+',default=['21-12','22-03','22-09'],help='months for the separate room-PUE comparison (need logics_pub)')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if '22-03' not in args.months:
        raise ValueError('22-03 is the calibration month and must be in --months')
    prov=provenance();prov['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov['months']=args.months;prov['facility_months']=args.facility_months
    (args.out/'provenance.json').write_text(json.dumps(prov,indent=2))
    panels={}; coverage={}
    for month in args.months:
        panels[month],coverage[month]=panel(args.data,args.cache,month,args.threads)
    model=fit(panels['22-03']);(args.out/'locked_march_parameters.json').write_text(json.dumps(model,indent=2))
    for month,h in panels.items():
        evaluate(h,model,args.out/month)
        if args.llnl:
            meter=load_llnl(str(args.llnl))
            if meter is None:raise FileNotFoundError(args.llnl)
            meter['hour']=(meter.hour.astype('int64')//3_600_000_000_000).astype('int64')
            c=pd.read_csv(args.out/month/'cluster_common_support.csv')
            c=c.merge(meter[['hour','kw']],on='hour',how='left',validate='one_to_one')
            c['switchboard_over_compute_psu']=c['kw']*1000/c.observed_psu
            c['complete_compute_fleet']=c.n_nodes.eq(980)
            c.to_csv(args.out/month/'switchboard_common_support.csv',index=False)
            good=c[c.complete_compute_fleet & c.kw.gt(0)]
            closure={'complete_hours':len(good),'scope':'980 compute nodes only; IT switchboard also includes service/network loads, no scaling of partial sums',
                     'switchboard_over_compute_psu_quantiles':good.switchboard_over_compute_psu.quantile([.05,.5,.95]).to_dict()}
            (args.out/month/'switchboard_summary.json').write_text(json.dumps(closure,indent=2))
    (args.out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    # Separate measured room seasonal comparison; July has no room meter.
    for month in args.facility_months:
        d=args.out/(month+'_facility');d.mkdir(exist_ok=True)
        logics=load_logics(str(args.data),month)
        if logics is None:raise FileNotFoundError(f'Required facility month missing: {month}')
        (d/'summary.json').write_text(json.dumps(check_c_pue(logics,None,str(d)),indent=2))
    print('COMPLETE: no controlled-response or ASI coefficient-transfer claim',flush=True)
if __name__=='__main__':main()
