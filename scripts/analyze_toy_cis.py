"""Fit every frozen toy surface, including flat axes and simple false positives."""
from pathlib import Path
import importlib.util,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/toy-cis-v1'
spec=importlib.util.spec_from_file_location('original_superellipse',ROOT/'vendor/fsec/scripts/lib/superellipse.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
P=json.loads((ROOT/'configs/toy-cis-v1.json').read_text());prepared=json.loads((OUT/'prepared.json').read_text())

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 return v

def write(path,v):path.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def fit(grid,fraction):
 chart=np.linspace(0,60,len(grid));axismax=[float(grid[:,0].max()),float(grid[0,:].max())]
 threshold=fraction*min(axismax)
 result={'threshold':threshold,'threshold_fraction':fraction,'axis_max_response':axismax,'p':None,'flags':[]}
 if threshold<1e-10:result['flags'].append('flat_axis');return result,None
 t=np.array([F.axis_intercept(chart,grid[:,0],threshold),F.axis_intercept(chart,grid[0,:],threshold)])
 if not np.isfinite(t).all() or (t<=0).any():result['flags'].append('invalid_intercept');return result,None
 raw=F.extract_contour(chart,grid,threshold)
 raw=raw[(raw[:,0]<59)&(raw[:,1]<59)]
 points=raw/t;mask=(points[:,0]>.05)&(points[:,1]>.05)&(points[:,0]<1.5)&(points[:,1]<1.5);points=points[mask]
 result.update(F.fit_superellipse(points[:,0],points[:,1]));result['intercepts_feature_units']=t/40
 if len(points)<3:result['flags'].append('insufficient_contour');return result,points
 x,y=points.T;a=x*x+y*y-1;b=2*x*y
 delta_raw=-float(np.sum(a*b)/np.sum(b*b));delta=float(np.clip(delta_raw,-1,1))
 residual=float(np.abs(np.sqrt(np.clip(x*x+2*delta*x*y+y*y,0,None))-1).mean())
 result['quadratic']={'cross_term':delta,'unconstrained_cross_term':delta_raw,'psd_boundary':abs(delta)>=1,
  'mean_radial_frac':residual,'better_than_superellipse':residual<result['mean_radial_frac']}
 if result['p']<.501 or result['p']>9.99:result['flags'].append('p_bound')
 if result['mean_radial_frac']>.02:result['flags'].append('superellipse_residual_over_2pct')
 if min(axismax)<.01:result['flags'].append('axis_response_below_0p01')
 if any((np.diff(g)<-1e-8).any() for g in [grid[:,0],grid[0,:]]):result['flags'].append('nonmonotone_axis')
 return result,points

def calibration():
 amp=np.linspace(0,1.5,81);a,b=np.meshgrid(amp,amp,indexing='ij');rows=[]
 functions={'orthogonal_linear_p2':lambda x,y:np.hypot(x,y),'tilted_linear':lambda x,y:np.sqrt(x*x-x*y+y*y),
  'separable_square_p4':lambda x,y:np.hypot(x*x,y*y),'dedicated_relu':lambda x,y:np.hypot(np.maximum(x,0),np.maximum(y,0))}
 arrays={};points={}
 for name,fn in functions.items():
  for sign in P['signs']:
   grid=fn(sign[0]*a,sign[1]*b);result,pts=fit(grid,.5);rows.append({'name':name,'sign':sign,**result})
   if sign==[1,1]:arrays[name]=grid;points[name]=pts
 for r in rows:
  if r['name']=='orthogonal_linear_p2':assert abs(r['p']-2)<.03,r
  if r['name']=='separable_square_p4':assert abs(r['p']-4)<.05,r
  if r['name']=='tilted_linear':assert r['quadratic']['mean_radial_frac']<.001,r
 np.savez_compressed(OUT/'analytic-controls.npz',**arrays)
 return rows,points

def main():
 start=time.perf_counter();raw=np.load(OUT/'surfaces.npz');meta=json.loads((OUT/'surfaces.json').read_text());records=[];post=[];fine=[]
 for index,(row,grids) in enumerate(zip(meta['rows'],raw['l2'])):
  median=np.median(grids,axis=0);fits={}
  for fraction in P['threshold_fractions']:fits[str(fraction)],_=fit(median,fraction)
  per=[fit(g,.5)[0] for g in grids]
  records.append({**row,'fits':fits,'per_background':per,'max_gate_switch_fraction':float(raw['gate_switch_fraction'][index].max())})
  if index%72==0:print('FIT',index,'/',len(meta['rows']),flush=True)
 for row,grids in zip(meta['post_rows'],raw['post_l2']):post.append({**row,'fit':fit(grids[0],.5)[0]})
 for row,grids in zip(meta['fine_rows'],raw['fine_l2']):
  result,_=fit(np.median(grids,axis=0),.5)
  coarse=next(r for r in records if all(r[k]==row[k] for k in ['model','pair','kind','sign','state']))['fits']['0.5']
  fine.append({**row,'fit':result,'coarse_p':coarse['p'],'delta_p':result['p']-coarse['p'] if result['p'] is not None and coarse['p'] is not None else None})
 calibrations,calpoints=calibration();cohorts=[]
 for name in P['models']:
  for state in P['anchor_pair_values']:
   for kind in P['direction_kinds']:
    selected=[r['fits']['0.5'] for r in records if r['model']==name and r['state']==state and r['kind']==kind]
    valid=[r for r in selected if r['p'] is not None and np.isfinite(r['p'])];values=[r['p'] for r in valid]
    cohorts.append({'model':name,'state':state,'kind':kind,'attempted':len(selected),'finite':len(valid),
     'median_p':float(np.median(values)) if values else None,'p_gt2':sum(v>2 for v in values),'flagged':sum(bool(r['flags']) for r in valid),
     'quadratic_better':sum(r['quadratic']['better_than_superellipse'] for r in valid if 'quadratic' in r),
     'median_superellipse_residual':float(np.median([r['mean_radial_frac'] for r in valid])) if valid else None,
     'median_quadratic_residual':float(np.median([r['quadratic']['mean_radial_frac'] for r in valid if 'quadratic' in r])) if valid else None})
 summary={'records':records,'post_relu':post,'fine_checks':fine,'analytic_controls':calibrations,'cohorts':cohorts,
  'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,
  'inference_raw_sha256':hashlib.sha256((OUT/'surfaces.npz').read_bytes()).hexdigest(),
  'scope':'Descriptive pilot; one checkpoint per training objective, six pairs selected using common single-feature eligibility, four matched backgrounds. No population p-values.'}
 write(OUT/'analysis.json',summary)
 colors={'L2':'#3b79a7','L4':'#3c856a','dedicated':'#a76c4b'};labels={'L2':'L2 checkpoint','L4':'L4 checkpoint','dedicated':'Dedicated ReLU'}
 fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
 for ax,state in zip(axes,P['anchor_pair_values']):
  for i,(name,kind) in enumerate((m,k) for m in P['models'] for k in P['direction_kinds']):
   selected=[r for r in records if r['model']==name and r['state']==state and r['kind']==kind];valid=[r for r in selected if r['fits']['0.5']['p'] is not None and np.isfinite(r['fits']['0.5']['p'])]
   vals=[r['fits']['0.5']['p'] for r in valid];jitter=np.linspace(-.16,.16,len(vals))
   ax.scatter(i+jitter,vals,s=17,color=colors[name],alpha=.75,marker='o' if kind=='codeword' else 'x')
   if vals:ax.plot([i-.22,i+.22],[np.median(vals)]*2,color='#222',lw=2)
   ax.text(i,.55,f'{len(vals)}/24',ha='center',fontsize=7,color='#777')
  ax.axhline(2,color='#999',ls='--',lw=1);ax.set(title=f'Pair inputs: {state} ({P["anchor_pair_values"][state]:+.1f})',xticks=range(6),
   xticklabels=[f'{m}\n'+('code' if k=='codeword' else 'random') for m in P['models'] for k in P['direction_kinds']],ylim=(.45,10.3),ylabel='Fitted p')
  ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 fig.suptitle('Toy pilot: geometry depends on activity, sign, and direction family\nEach group attempts six pairs × four signs; all finite fits retained, including flagged fits. Black marks: medians.',fontsize=12)
 fig.tight_layout();fig.savefig(OUT/'geometry.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,9))
 for name in P['models']:
  err=np.sort(prepared['models'][name]['per_feature_positive_active_mse']);axes[0,0].plot(np.arange(100),err,label=labels[name],color=colors[name])
 axes[0,0].set(title='Clean task: per-feature error, sorted within each model',xlabel='Feature rank',ylabel='Positive-active MSE');axes[0,0].legend(fontsize=8)
 for name in ['orthogonal_linear_p2','tilted_linear','separable_square_p4']:
  r=next(r for r in calibrations if r['name']==name and r['sign']==[1,1]);pts=calpoints[name]
  axes[0,1].plot(pts[:,0],pts[:,1],label=f'{name.replace("_"," ")}: p={r["p"]:.2f}')
 axes[0,1].set(title='Known mechanisms: nonlinear-looking contours can mislead',xlabel='Normalized first coefficient',ylabel='Normalized second coefficient',aspect='equal');axes[0,1].legend(fontsize=7)
 corruption=json.loads((OUT/'corruption.json').read_text())['records']
 for r in corruption:r['common_eligible_mse']=float(np.mean(np.asarray(r['per_feature_positive_active_mse'])[prepared['common_eligible_features']]))
 styles={'natural_relu':'-','frozen_gates':'--','decoder_only':':'}
 for ax,metric,title in [(axes[1,0],'common_eligible_mse','Fixed-target corruption: shared eligible features'),(axes[1,1],'inactive_false_positive_rate_above_0p1','Fixed-target corruption: inactive false positives')]:
  for name in P['models']:
   for method,style in styles.items():
    selected=[r for r in corruption if r['model']==name and r['method']==method];ax.plot([r['level'] for r in selected],[r[metric] for r in selected],color=colors[name],ls=style)
  ax.set(title=title,xlabel='Noise norm / median encoder-column norm',ylabel='MSE' if metric.endswith('mse') else 'Fraction predicted > 0.1')
 handles=[Line2D([0],[0],color=colors[n],label=labels[n]) for n in P['models']]+[Line2D([0],[0],color='#555',ls=s,label=m.replace('_',' ')) for m,s in styles.items()]
 axes[1,0].legend(handles=handles,fontsize=7,ncol=2)
 for ax in axes.ravel():ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
 fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,8))
 for col,name in enumerate(P['models']):
  for rowi,state in enumerate(['inactive','negative']):
   ax=axes[rowi,col]
   for kind,style in [('codeword','-'),('matched_random','--')]:
    idx=next(i for i,r in enumerate(meta['rows']) if r['model']==name and r['pair']==0 and r['state']==state and r['kind']==kind and r['sign']==[1,1])
    result,points=fit(np.median(raw['l2'][idx],axis=0),.5)
    if points is not None and len(points):ax.plot(points[:,0],points[:,1],style,label=f'{kind.replace("_"," ")}: p={result["p"]:.2f}')
   x,y=F.superellipse_curve(2);ax.plot(x,y,':',color='#aaa');ax.set(title=f'{labels[name]}, {state}',xlabel='Normalized first coefficient',ylabel='Normalized second coefficient',aspect='equal',xlim=(0,1.5),ylim=(0,1.5));ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('First frozen feature pair: same ++ sign, different anchor activity\nDotted curve: axis-aligned p=2 reference; additive interventions.',fontsize=12);fig.tight_layout();fig.savefig(OUT/'contours.png',dpi=170,bbox_inches='tight');plt.close(fig)
 lines=['model,state,kind,attempted,finite,median_p,p_gt2,flagged,quadratic_better']
 for r in cohorts:lines.append(','.join(str(r[k]) for k in lines[0].split(',')))
 (OUT/'cohorts.csv').write_text('\n'.join(lines)+'\n')
 print('COMPLETE',len(records),'primary fits;',len(post),'post-ReLU controls;',len(fine),'fine-grid checks',flush=True)

if __name__=='__main__':main()
