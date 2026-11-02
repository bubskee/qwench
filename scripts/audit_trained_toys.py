"""Final audit: all networks actually trained; correct task/cohorts/geometry provenance."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import train_toy_activations as T

def main():
 out=T.OUT;done=json.loads((out/'TRAINING-COMPLETE.json').read_text());assert done['step']==100000 and done['networks']==30
 seal=json.loads((out/'input-seal.json').read_text());assert done['dependencies']=={k:seal[k] for k in ['source_sha256','protocol_sha256','dataset_sha256']}
 assert seal['source_sha256']==T.sha(T.__file__) and seal['protocol_sha256']==T.sha(T.ROOT/'configs/toy-trained-v1.json')
 final=load_file(str(out/'weights-100000.safetensors'));assert T.sha(out/'weights-100000.safetensors')==done['final_weights_sha256']
 logs=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
 assert logs[0]['step']==1 and logs[-1]['step']==100000 and logs[-1]['lr']==0
 assert [r['step'] for r in logs]==[1]+list(range(1000,100001,1000))
 assert all(np.isfinite(v).all() for r in logs for v in r['loss_by_architecture'].values())
 initialized,_=T.initialize('cpu');updates=[]
 for a in T.P['architectures']:
  roles=['gate','out']+(['value'] if a['activation']=='swiglu' else [])
  for index,(exponent,seed) in enumerate(T.network_ids()):
   norms={role:float((final[f'{a["name"]}.{role}'][index]-initialized[a['name']][role][index].detach()).norm()) for role in roles}
   assert min(norms.values())>1e-8
   updates.append({'id':f'{a["name"]}_L{exponent}_s{seed}','parameter_change_norms':norms})
 evaluation=json.loads((out/'eval-100000.json').read_text())['networks'];assert len(evaluation)==30
 plan=json.loads((out/'geometry-plan.json').read_text());inputs=np.load(out/'geometry-inputs.npz');cases=0;mainfits=0;flat=0;flagged=0;fine_deltas=[]
 for row,net in zip(evaluation,plan['networks']):
  assert row['id']==net['id'];eligible=set(row['competent_feature_indices']);expected=[]
  for pair in inputs['pair_order']:
   if all(int(j) in eligible for j in pair):expected.append(pair.tolist())
   if len(expected)==3:break
  assert expected==net['conditioned_pairs']
  source=out/'surfaces'/(net['id']+'.npz');meta=json.loads(source.with_suffix('.json').read_text());data=np.load(source)
  assert T.sha(source)==meta['raw_sha256'] and T.sha(out/'geometry-plan.json')==meta['plan_sha256']
  for key in ['l2','fine_l2']:
   assert np.isfinite(data[key]).all() and (data[key]>=0).all()
   if data[key].size:assert data[key][...,0,0].max()<1e-9
   cases+=data[key].size
  for item in meta['rows']:
   assert item['competent']==all(j in eligible for j in item['features'])
  fits=json.loads((out/'analysis'/(net['id']+'.json')).read_text());assert fits['dependencies']['raw_sha256']==T.sha(source)
  assert len(fits['rows'])==len(meta['rows']);mainfits+=len(fits['rows'])
  for f in fits['rows']:
   fit=f['fits']['0.5'];flat+=fit['p'] is None;flagged+=bool(fit['flags']) and fit['p'] is not None
  fine_deltas.extend(abs(r['delta_p']) for r in fits['fine'] if r['delta_p'] is not None)
 finalcheck=json.loads((out/'implementation-checks-100000.json').read_text());assert finalcheck['all_assertions_passed'] and len(finalcheck['networks'])==30
 result={'all_assertions_passed':True,'trained_networks':30,'updates_each':100000,'loss_log_records':len(logs),'all_parameters_changed_from_initialization':True,
  'parameter_changes':updates,'perturbation_response_points':cases,'primary_aggregate_surfaces':mainfits,'invalid_primary_fits':flat,'flagged_finite_primary_fits':flagged,
  'fine_max_abs_delta_p':max(fine_deltas) if fine_deltas else None,'full_swiglu_has_independently_trained_gate_and_value':True,
  'scope':'Numerical/artifact/training checks, not validation of a universal CiS metric or an error-correction mechanism.'}
 T.write(out/'quality-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='parameter_changes'},indent=2))

if __name__=='__main__':main()
