"""Conditional baseline power envelope; NOT controlled replay or deliverability proof."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import powermodel as pm
from provenance import provenance

def windows(x,h):
    w=np.lib.stride_tricks.sliding_window_view(x,h)
    return w[np.isfinite(w).all(axis=1)].min(axis=1)

def cap(x,a):
    # Exact largest observed threshold satisfying empirical coverage >= a.
    return float(np.sort(x)[len(x)-int(np.ceil(a*len(x)))])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--agg',required=True); ap.add_argument('--a1',required=True); ap.add_argument('--configs',required=True); ap.add_argument('--out',required=True)
    ar=ap.parse_args(); out=Path(ar.out); out.mkdir(parents=True,exist_ok=True)
    pod=pd.read_parquet(ar.agg); pod['t']=pod.day.astype(int)*24+pod.hour.astype(int)
    t=pod.t.to_numpy(); nt=t.max()+1; codes,uniq=pd.factorize(pod.gpu_spec_public); specs=sorted(pod.gpu_spec_public.dropna().unique())
    layer=pm.assign_layer(pod,'central'); masks={'lp':layer=='curtail','training':(layer=='shift')&(pod.job_type_public.to_numpy()=='training'),'offline':(layer=='shift')&(pod.job_type_public.to_numpy()=='offline_inference')}
    assert not np.any(masks['lp'] & (masks['training'] | masks['offline']))
    assert np.all((masks['training']|masks['offline'])==(layer=='shift'))
    a1=pd.read_parquet(ar.a1).set_index('t').reindex(range(nt)); good=a1['have'].to_numpy(bool)
    cfg_paths=[Path(ar.configs)/f for f in ['power_curves.yaml','power_curves_a7_measured_low.yaml','power_curves_a7_measured_mid.yaml']]
    cases={}; rows=[]; summaries={}; checks={}
    for cp in cfg_paths:
        cfg=yaml.safe_load(cp.read_text()); p=pm.draw_params(cfg,np.random.default_rng(0),specs,base=True)
        parts=pm.row_energy(pod,cfg,p,codes,uniq)
        ss={}
        for cat,m in masks.items():
            for part in ['active','idle','total']:
                x=np.bincount(t[m],weights=parts[part][m]/1e6,minlength=nt); x[~good]=np.nan; ss[cat+'_'+part]=x
        ss['workload']=np.bincount(t,weights=parts['total']/1e6,minlength=nt); ss['workload'][~good]=np.nan
        key=cp.stem; cases[key]=ss
        summaries[key]={k:float(np.nanmean(v)) for k,v in ss.items()}
        for c in masks:
            assert np.allclose(ss[c+'_total'][good],(ss[c+'_active']+ss[c+'_idle'])[good])
        for at in [0,.25,.5,.75,1]:
            for ao in [0,.25,.5,.75,1]:
                sh=at*ss['training_active']+ao*ss['offline_active']; combo=ss['lp_active']+sh
                for h in [1,4,24]:
                    w=windows(combo,h); sw=windows(sh,h)
                    for a in [.5,.8,.9,.95,.99]:
                        k=cap(w,a); ks=cap(sw,a)
                        rows.append(dict(model=key,training_fraction=at,offline_fraction=ao,hours=h,coverage=a,n_windows=len(w),combined_mw=k,incremental_shift_mw=ks,combined_actual_coverage=float(np.mean(w>=k)),combined_linear_quantile_mw=float(np.quantile(w,1-a))))
    base=cases['power_curves']; tbl=pd.DataFrame(rows); tbl.to_csv(out/'capacity_grid.csv',index=False)
    pd.DataFrame({'t':range(nt),'good':good,**base,'facility_base_mw':a1.base_mw,'facility_p50_mw':a1.p50_mw}).to_csv(out/'hourly_components.csv',index=False)
    # Same marginal convention as manuscript: GPU active MW equals facility relief
    # for marginal PUE=1, dynamic host reduction excluded. Baseline base denominator.
    total=base['lp_active']+base['training_active']+base['offline_active']
    target_rows=[]
    for denom_name in ['base_mw','p50_mw']:
        f=a1[denom_name].to_numpy(copy=True)
        # A1 retains a partial placeholder at missing day5/hour6. Exclude it
        # from facility thresholds just as it is excluded from response windows.
        f[~good]=np.nan
        for marg in [1.,1.2]:
            for fraction in [.2,.3,.5,1.]:
                for h in [1,4,24]:
                    margin=windows(marg*total-fraction*f,h)
                    target_rows.append(dict(facility_denominator=denom_name,marginal_pue=marg,target_fraction=fraction,hours=h,n_windows=len(margin),fraction_windows_envelope_covers_target=float(np.mean(margin>=0)), full_envelope_k95_mw=cap(windows(marg*total,h),.95), fraction_facility_k95_mw=cap(windows(fraction*f,h),.95), same_k95_ratio=cap(windows(marg*total,h),.95)/cap(windows(fraction*f,h),.95)))
    pd.DataFrame(target_rows).to_csv(out/'facility_targets.csv',index=False)
    # Restoring deferred work with same active energy efficiency. e is added active
    # energy fraction (recomputation/other), H incremental usable active power.
    rec=[]
    for _,r in tbl[(tbl.model=='power_curves')&(tbl.coverage==.95)&(tbl.training_fraction==tbl.offline_fraction)&(tbl.training_fraction>0)].iterrows():
        for e in [0,.05,.1,.25]:
            for H in [.5,1,2,5]:
                E=r.incremental_shift_mw*r.hours
                rec.append(dict(training_fraction=r.training_fraction,offline_fraction=r.offline_fraction,hours=r.hours,coverage=r.coverage,shift_envelope_mw=r.incremental_shift_mw,deferred_active_mwh=E,extra_energy_fraction=e,recovery_active_mwh=(1+e)*E,external_recovery_headroom_mw=H,recovery_time_lower_bound_h=(1+e)*E/H,net_active_energy_saved_mwh=-e*E))
    recovery=pd.DataFrame(rec); recovery.to_csv(out/'recovery_scenarios.csv',index=False)
    checks['hours_valid']=int(good.sum()); checks['hours_total']=int(nt)
    checks['window_counts']={str(h):len(windows(total,h)) for h in [1,4,24]}
    checks['all_coverage_thresholds_attained']=bool((tbl.combined_actual_coverage+1e-12>=tbl.coverage).all())
    # nonnegative parts and monotone capacities; do not bridge missing slots
    assert checks['all_coverage_thresholds_attained']
    assert all(np.all(v[good]>=0) for k,v in base.items())
    for _,g in tbl.groupby(['model','offline_fraction','hours','coverage']): assert np.all(np.diff(g.sort_values('training_fraction').combined_mw)>=-1e-12)
    for _,g in tbl.groupby(['model','training_fraction','hours','coverage']): assert np.all(np.diff(g.sort_values('offline_fraction').combined_mw)>=-1e-12)
    checks['monotonic_participation']=True
    for _,g in tbl.groupby(['model','training_fraction','offline_fraction','hours']):
        assert np.all(np.diff(g.sort_values('coverage').combined_mw)<=1e-12)
    # Equal-start toy isolates duration monotonicity from finite-sample edge effects.
    toy=np.array([2.,5.,1.,4.,3.,6.]); starts=np.arange(4)
    mm=[np.array([toy[i:i+h].min() for i in starts]) for h in [1,2,3]]
    assert np.all(mm[0]>=mm[1]) and np.all(mm[1]>=mm[2])
    assert np.all(np.diff([cap(mm[0],a) for a in [.5,.8,1.]])<=0)
    xx=np.array([0.,10.]); yy=xx[::-1]
    assert cap(xx+yy,.95)!=cap(xx,.95)+cap(yy,.95)
    assert len(windows(np.array([1.,np.nan,2.,3.]),2))==1
    checks['coverage_monotonic_and_duration_equal_start_toy']=True
    checks['quantile_nonadditivity_toy']=True
    checks['gap_exclusion_toy']=True
    # Compare original all-fleet A5 baseline if supplied through sibling path.
    orig=Path(ar.a1).parent.parent/'a5/envelope_hourly.parquet'
    if orig.exists():
        e=pd.read_parquet(orig).set_index('t').reindex(range(nt))
        checks['max_lp_difference_from_saved_a5']=float(np.nanmax(np.abs(base['lp_active']-e.curtail_idle_retained.to_numpy())))
        assert checks['max_lp_difference_from_saved_a5']<1e-8
    summary={'mean_components_mw':summaries,'facility_mean_base_mw':float(a1.loc[good,'base_mw'].mean()),'facility_mean_p50_mw':float(a1.loc[good,'p50_mw'].mean()),'checks':checks,'marginal_pue_default':1.0,'interpretation':'Conditional baseline envelope; not operational capacity. Idle retained; no host reduction. No backfill; fractional divisible control assumed. Unknown checkpoint, SLA, job identity continuity and recovery constraints.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pr=provenance(); pr['inputs']={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [ar.agg,ar.a1,*cfg_paths,Path(__file__),Path(pm.__file__)]}; pr['command']=sys.argv; pr['numpy']=np.__version__; pr['pandas']=pd.__version__; (out/'provenance.json').write_text(json.dumps(pr,indent=2))
    plot(out,base,tbl,recovery,good)
    print(json.dumps(summary,indent=2))

def plot(out,base,tbl,recovery,good):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#222222','text.color':'#222222'})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for cat,lab,col,ls in [('lp','LP active','#555555',':'),('training','HP training active','#2471A3','-'),('offline','HP offline inference active','#C17C20','--')]:
        vals=base[cat+'_active'][good]; x=np.unique(np.r_[0,vals]); y=(len(vals)-np.searchsorted(np.sort(vals),x,side='left'))/len(vals)*100
        axs[0].step(x,y,where='pre',label=lab,color=col,ls=ls,lw=1.6)
    axs[0].set(xlabel='Active GPU power threshold (MW)',ylabel='Historical hours meeting threshold (%)',xlim=(0,None),ylim=(0,101),title='One-hour baseline power by category'); axs[0].legend(frameon=False,fontsize=9)
    g=tbl[(tbl.model=='power_curves')&(tbl.hours==4)&(tbl.coverage==.95)].pivot(index='offline_fraction',columns='training_fraction',values='combined_mw')
    im=axs[1].imshow(g,origin='lower',vmin=0,cmap='Blues',aspect='auto')
    for i in range(5):
        for j in range(5): axs[1].text(j,i,f'{g.iloc[i,j]:.2f}',ha='center',va='center',color='white' if g.iloc[i,j]>g.to_numpy().max()*.65 else '#222222')
    axs[1].set_xticks(range(5),['0','25','50','75','100']); axs[1].set_yticks(range(5),['0','25','50','75','100']); axs[1].set(xlabel='Assumed HP training active-power participation (%)',ylabel='Assumed HP offline participation (%)',title='LP + conditional shift envelope (MW)\n4-hour window; 95% historical coverage')
    fig.suptitle('Conditional power envelopes with GPU idle power retained',fontsize=13)
    fig.savefig(out/'f1_conditional_envelopes.png',dpi=180);fig.savefig(out/'f1_conditional_envelopes.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    g=recovery[(recovery.hours==4)&(recovery.training_fraction==1)]
    for H,col,ls in [(1,'#555555',':'),(2,'#2471A3','-'),(5,'#C17C20','--')]:
        z=g[g.external_recovery_headroom_mw==H]; ax.plot(z.extra_energy_fraction*100,z.recovery_time_lower_bound_h,label=f'{H} MW external recovery headroom',color=col,ls=ls,marker='o',lw=1.5)
    ax.set(xlabel='Assumed added recovery energy (%)',ylabel='Recovery time lower bound (h)',xlim=(0,25),ylim=(0,None),title='Recovery energy balance for the shift-only envelope\nFull HP participation; 4 h; 95% baseline coverage')
    ax.legend(frameon=False,fontsize=9,loc='center right');fig.savefig(out/'f2_recovery_bound.png',dpi=180);fig.savefig(out/'f2_recovery_bound.pdf');plt.close(fig)
if __name__=='__main__': main()
