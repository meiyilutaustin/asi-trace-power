#!/usr/bin/env python
"""Required workload share p(h,a) = K_C/K_L over the duration-reliability grid, plus a
tail-calibrated scalar, from the fleet envelope (usage: a6_required_share.py <products_dir> <out.json>)."""
import pandas as pd, numpy as np, json, sys
R = sys.argv[1] if len(sys.argv) > 1 else 'data/products'   # directory holding envelope_hourly.parquet
e=pd.read_parquet(f'{R}/envelope_hourly.parquet').sort_values('t')
c=e['curtail_idle_retained'].where(e['good']).to_numpy(); L=e['workload'].where(e['good']).to_numpy(); t=e['t'].to_numpy()
def runmin(x,h):
    n=len(x); out=np.full(n,np.nan)
    for i in range(n-h+1):
        w=x[i:i+h]
        if not np.isnan(w).any(): out[i]=w.min()
    return out
def K(x,h,a):
    m=runmin(x,h); m=m[~np.isnan(m)]; return float(np.quantile(m,1-a))
out={'K95':{h:K(c,h,.95) for h in (1,4,24)},'mean':float(np.nanmean(c))}
pm=np.nanmean(c)/np.nanmean(L); p=K(c,4,.95)/K(L,4,.95)
out['mean_share']=float(pm); out['tail_share_4h95']=float(p)
out['tail_cal_ratio']={h:p*K(L,h,.95)/K(c,h,.95) for h in (1,4,24)}
out['mean_cal_ratio']={h:pm*K(L,h,.95)/K(c,h,.95) for h in (1,4,24)}
out['required_share_over_mean']={h:{a:(K(c,h,a)/K(L,h,a))/pm for a in (.5,.9,.95,.99)} for h in (1,2,4,8,24)}
hod=t%24; ph=(t//24)%7
for h in (1,4):
    m=runmin(c,h)
    out[f'K95_h{h}_by_hour']=[float(np.quantile(m[(hod==k)&~np.isnan(m)],.05)) for k in range(24)]
    out[f'K95_h{h}_by_phase']=[float(np.quantile(m[(ph==k)&~np.isnan(m)],.05)) for k in range(7)]
    out[f'K95_h{h}_by_30d']=[float(np.quantile(m[(t//24//30==k)&~np.isnan(m)],.05)) for k in range(7)]
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else 'required_share.json','w'),indent=1)
