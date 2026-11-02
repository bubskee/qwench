"""Fit all final trained-toy contours, keeping fixed and competent cohorts separate."""
from pathlib import Path
import json,time
import numpy as np
import analyze_toy_cis as A
import train_toy_activations as T
OUT=T.OUT;AN=OUT/'analysis'

def main():
 assert (OUT/'GEOMETRY-COMPLETE.json').exists();AN.mkdir(exist_ok=True);plan=json.loads((OUT/'geometry-plan.json').read_text());started=time.perf_counter();allcohorts=[]
 for network in plan['networks']:
  ident=network['id'];source=OUT/'surfaces'/(ident+'.npz');meta=json.loads(source.with_suffix('.json').read_text());target=AN/(ident+'.json')
  signature={'raw_sha256':T.sha(source),'source_sha256':T.sha(__file__),'fit_helper_sha256':T.sha(A.__file__)}
  if target.exists():
   record=json.loads(target.read_text());assert record['dependencies']==signature
  else:
   data=np.load(source);rows=[];fine=[]
   for row,grids in zip(meta['rows'],data['l2']):
    median=np.median(grids,axis=0);fits={str(f):A.fit(median,f)[0] for f in T.P['geometry']['threshold_fractions']}
    rows.append({**row,'fits':fits,'per_background':[A.fit(g,.5)[0] for g in grids]})
   for row,grids in zip(meta['fine_rows'],data['fine_l2']):
    result,_=A.fit(np.median(grids,axis=0),.5)
    coarse=next(r['fits']['0.5'] for r in rows if all(r[k]==row[k] for k in ['pair_index','kind','anchor_value','sign']))
    delta=result['p']-coarse['p'] if result['p'] is not None and coarse['p'] is not None else None
    fine.append({**row,'fit':result,'coarse_p':coarse['p'],'delta_p':delta})
   cohorts=[]
   for cohort in ['fixed','fixed_competent','competence_conditioned']:
    for kind in T.P['geometry']['direction_kinds']:
     for anchor in T.P['geometry']['pair_anchor_values']:
      for quadrant in ['++','all']:
       selected=[]
       for r in rows:
        if r['kind']!=kind or r['anchor_value']!=anchor or (quadrant=='++' and r['sign']!=[1,1]):continue
        if cohort=='fixed_competent':accepted='fixed' in r['cohorts'] and r['competent']
        else:accepted=cohort in r['cohorts']
        if accepted:selected.append(r)
       valid=[r['fits']['0.5'] for r in selected if r['fits']['0.5']['p'] is not None and np.isfinite(r['fits']['0.5']['p'])]
       cohorts.append({'cohort':cohort,'kind':kind,'anchor_value':anchor,'quadrant':quadrant,'attempted':len(selected),'finite':len(valid),
        'median_p':float(np.median([r['p'] for r in valid])) if valid else None,'flagged':sum(bool(r['flags']) for r in valid),
        'median_radial_residual':float(np.median([r['mean_radial_frac'] for r in valid])) if valid else None,
        'quadratic_better':sum(r['quadratic']['better_than_superellipse'] for r in valid if 'quadratic' in r)})
   record={'network':network,'rows':rows,'fine':fine,'cohorts':cohorts,'dependencies':signature};A.write(target,record)
  allcohorts.extend([{**network,**r} for r in record['cohorts']]);print('FIT_COMPLETE',ident,'elapsed seconds',round(time.perf_counter()-started,1),flush=True)
 A.write(AN/'cohorts.json',allcohorts)
 keys=['id','architecture','loss_exponent','seed','cohort','kind','anchor_value','quadrant','attempted','finite','median_p','flagged','median_radial_residual','quadratic_better']
 (AN/'cohorts.csv').write_text(','.join(keys)+'\n'+'\n'.join(','.join(str(r[k]) for k in keys) for r in allcohorts)+'\n')
 T.write(OUT/'ANALYSIS-COMPLETE.json',{'networks':len(plan['networks']),'seconds':time.perf_counter()-started,'source_sha256':T.sha(__file__)})

if __name__=='__main__':main()
