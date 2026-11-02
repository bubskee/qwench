"""Final audit: all networks actually trained; correct task/cohorts/geometry provenance."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import importlib,os
T=importlib.import_module(os.environ.get('QWENCH_TRAINED_DRIVER','train_scalar_core'))

def verify_training(module):
 out=module.OUT;done=json.loads((out/'TRAINING-COMPLETE.json').read_text());expected=len(module.P['architectures'])*len(module.network_ids())
 assert done['step']==100000 and done['networks']==expected
 seal=json.loads((out/'input-seal.json').read_text());assert done['dependencies']=={k:seal[k] for k in ['source_sha256','protocol_sha256','dataset_sha256']}
 assert seal['source_sha256']==module.sha(module.__file__) and seal['protocol_sha256']==module.sha(module.ROOT/'configs'/(module.P['version']+'.json'))
 final=load_file(str(out/'weights-100000.safetensors'));assert module.sha(out/'weights-100000.safetensors')==done['final_weights_sha256']
 logs=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
 assert [r['step'] for r in logs]==[1]+list(range(1000,100001,1000)) and logs[-1]['lr']==0
 assert all(np.isfinite(v).all() for r in logs for v in r['loss_by_architecture'].values())
 initialized,_=module.initialize('cpu');updates=[]
 for arch in module.P['architectures']:
  roles=['gate','out']+(['value'] if arch['activation']=='swiglu' else [])
  for index,(exponent,seed) in enumerate(module.network_ids()):
   norms={role:float((final[f'{arch["name"]}.{role}'][index]-initialized[arch['name']][role][index].detach()).norm()) for role in roles}
   assert min(norms.values())>1e-8
   updates.append({'id':f'{arch["name"]}_L{exponent}_s{seed}','parameter_change_norms':norms})
 return updates,{'run':module.P['version'],'networks':expected,'log_records':len(logs),'dataset_sha256':seal['dataset_sha256'],'completion_sha256':module.sha(out/'TRAINING-COMPLETE.json')},final

def main():
 out=T.OUT;done=json.loads((out/'TRAINING-COMPLETE.json').read_text());updates=[];proofs=[]
 if 'source_cohorts' in T.P:
  merged=load_file(str(out/'weights-100000.safetensors'))
  for name in T.P['source_cohorts']:
   module=importlib.import_module(name);changes,proof,weights=verify_training(module);updates+=changes;proofs.append(proof)
   for key,value in weights.items():assert torch.equal(value,merged[key])
  assert len({p['dataset_sha256'] for p in proofs})==1
  assert done['networks']==sum(p['networks'] for p in proofs)==30
 else:
  updates,proof,weights=verify_training(T);proofs.append(proof)
 evaluation=json.loads((out/'eval-100000.json').read_text())['networks'];assert len(evaluation)==len(T.P['architectures'])*len(T.network_ids())
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
 finalcheck=json.loads((out/'implementation-checks-100000.json').read_text());assert finalcheck['all_assertions_passed'] and len(finalcheck['networks'])==len(evaluation)
 result={'all_assertions_passed':True,'trained_networks':len(evaluation),'updates_each':100000,'source_training_proofs':proofs,'all_parameters_changed_from_initialization':True,
  'parameter_changes':updates,'perturbation_response_points':cases,'primary_aggregate_surfaces':mainfits,'invalid_primary_fits':flat,'flagged_finite_primary_fits':flagged,
  'fine_max_abs_delta_p':max(fine_deltas) if fine_deltas else None,'scalar_activations_trained_from_scratch_or_exact_checkpoint_continuation':True,
  'scope':'Numerical/artifact/training checks, not validation of a universal CiS metric or an error-correction mechanism.'}
 T.write(out/'quality-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='parameter_changes'},indent=2))

if __name__=='__main__':main()
