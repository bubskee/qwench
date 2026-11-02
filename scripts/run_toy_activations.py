"""Frozen-weight activation ablations on sealed toy-CiS inputs; no training."""
from pathlib import Path
import hashlib,json,time,subprocess
import numpy as np
import torch
import torch.nn.functional as F
import run_toy_cis as B
ROOT=B.ROOT;OUT=ROOT/'reports/toy-activations-v1';P=json.loads((ROOT/'configs/toy-activations-v1.json').read_text())

def phi(z,name):
 if name=='relu':return z.relu()
 if name=='gelu':return F.gelu(z,approximate='none')
 if name=='silu':return F.silu(z)
 if name=='swiglu_tied':return F.silu(z)*z
 raise ValueError(name)

def grid(wi,wo,backgrounds,pair,value,directions,sign,n,activation):
 amp=np.linspace(0,1.5,n);a,b=np.meshgrid(amp,amp,indexing='ij')
 delta=B.tensor(np.stack([sign[0]*a.ravel(),sign[1]*b.ravel()],axis=1))@B.tensor(directions)
 x=B.tensor(backgrounds.copy());x[:,pair]=value;z=x@wi.T;clean=phi(z,activation)@wo.T
 return np.stack([(phi(anchor[None]+delta,activation)@wo.T-ref).norm(dim=-1).numpy().reshape(n,n) for anchor,ref in zip(z,clean)])

def metrics(x,prediction,features):
 target=x.relu();error=(prediction-target).square();positive=x>0;inactive=x==0;negative=x<0
 per=(error*positive).sum(0)/positive.sum(0).clamp_min(1)
 return {'positive_active_mse':float(error[positive].mean()),'common_eligible_mse':float(per[features].mean()),
  'inactive_rmse':float(error[inactive].mean().sqrt()),'negative_input_rmse':float(error[negative].mean().sqrt()),
  'inactive_false_positive_rate_above_0p1':float((prediction[inactive]>.1).double().mean()),'per_feature_positive_active_mse':per.tolist()}

def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'surfaces.npz').exists(),'Do not overwrite completed measurements'
 prepared,inputs=B.load_inputs();bank=B.models();meta=json.loads((B.OUT/'surfaces.json').read_text());parent=np.load(B.OUT/'surfaces.npz')
 deps=[Path(__file__),ROOT/'scripts/run_toy_cis.py',ROOT/'configs/toy-activations-v1.json',B.OUT/'prepared.json',B.OUT/'inputs.npz',B.OUT/'surfaces.npz']
 provenance={str(p.relative_to(ROOT)):B.sha(p) for p in deps}
 baseline_checks=[]
 for index in [0,23,144,167,288,311]:
  r=meta['rows'][index];pair=prepared['pairs'][r['pair']];wi,wo=bank[r['model']]
  computed=grid(wi,wo,inputs['backgrounds'],pair['features'],B.P['anchor_pair_values'][r['state']],inputs[f'{r["model"]}_{r["pair"]}_{r["kind"]}'],r['sign'],41,'relu')
  error=float(np.abs(computed-parent['l2'][index]).max());assert error<1e-12
  baseline_checks.append({'index':index,'max_abs':error})
 arrays={};start=time.perf_counter()
 for activation in P['activations'][1:]:
  rows=[]
  for r in meta['rows']:
   wi,wo=bank[r['model']];pair=prepared['pairs'][r['pair']]
   value=B.P['anchor_pair_values'][r['state']];directions=inputs[f'{r["model"]}_{r["pair"]}_{r["kind"]}']
   values=grid(wi,wo,inputs['backgrounds'],pair['features'],value,directions,r['sign'],41,activation)
   assert np.isfinite(values).all() and np.max(values[:,0,0])<1e-12
   rows.append(values)
  arrays[activation]=np.stack(rows);print('GEOMETRY',activation,'seconds',time.perf_counter()-start,flush=True)
 finerows=[]
 for name,(wi,wo) in bank.items():
  for state in ['inactive','negative']:
   for kind in B.P['direction_kinds']:finerows.append({'model':name,'pair':0,'state':state,'kind':kind,'sign':[1,1]})
 for activation in P['activations']:
  fine=[]
  for r in finerows:
   wi,wo=bank[r['model']];pair=prepared['pairs'][0];directions=inputs[f'{r["model"]}_0_{r["kind"]}']
   fine.append(grid(wi,wo,inputs['backgrounds'],pair['features'],B.P['anchor_pair_values'][r['state']],directions,[1,1],81,activation))
  arrays['fine_'+activation]=np.stack(fine)
 np.savez_compressed(OUT/'surfaces.npz',**arrays)
 B.write(OUT/'surfaces.json',{'rows':meta['rows'],'fine_rows':finerows,'activation_definitions':P['definitions'],'seconds':time.perf_counter()-start,
  'dependencies':provenance,'raw_sha256':B.sha(OUT/'surfaces.npz'),'relu_array_source':'../toy-cis-v1/surfaces.npz:l2','relu_grid_parity':baseline_checks})
 evaluation=B.sparse(B.P['evaluation_samples'],B.P['seed']);task=[]
 for name,(wi,wo) in bank.items():
  z=evaluation@wi.T
  for activation in P['activations']:
   single=phi(wi.T,activation)@wo.T;gain=single.diag();eligible=(gain>=.25)&(single.argmax(-1)==torch.arange(100))&(wi.norm(dim=0)>1e-8)
   row={'model':name,'activation':activation,'single_feature_self_gains':gain.tolist(),'eligible_features_after_swap':torch.where(eligible)[0].tolist(),
    'eligible_count_after_swap':int(eligible.sum()),**metrics(evaluation,phi(z,activation)@wo.T,prepared['common_eligible_features'])}
   if activation=='relu':assert abs(row['positive_active_mse']-prepared['models'][name]['positive_active_mse'])<1e-14
   task.append(row)
 noise_x=B.sparse(B.P['corruption_samples'],B.P['seed']+2);rng=np.random.default_rng(B.P['seed']+3);raw=B.tensor(rng.normal(size=(len(noise_x),100)));noise=[]
 prior_noise=json.loads((B.OUT/'corruption.json').read_text())['records']
 for name,(wi,wo) in bank.items():
  z=noise_x@wi.T;unit=raw[:,:len(wi)];unit=unit/unit.norm(dim=-1,keepdim=True);scale=prepared['models'][name]['median_encoder_column_norm']
  for activation in P['activations']:
   for level in B.P['corruption_levels']:
    prediction=phi(z+unit*(level*scale),activation)@wo.T
    row={'model':name,'activation':activation,'level':level,**metrics(noise_x,prediction,prepared['common_eligible_features'])}
    if activation=='relu':
     old=next(r for r in prior_noise if r['model']==name and r['method']=='natural_relu' and r['level']==level)
     for key in ['positive_active_mse','inactive_rmse','inactive_false_positive_rate_above_0p1']:assert row[key]==old[key],(name,level,key)
    noise.append(row)
 B.write(OUT/'task-and-noise.json',{'clean_task':task,'noise':noise,'target':'Original ReLU(clean input) for every activation; no target or weight adaptation.',
  'eval_input_sha256':hashlib.sha256(evaluation.numpy().tobytes()).hexdigest(),'corruption_input_sha256':hashlib.sha256(noise_x.numpy().tobytes()).hexdigest(),
  'dependencies':provenance,'source_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
 print('COMPLETE',len(meta['rows'])*3,'new aggregate surfaces;',len(finerows)*4,'fine grids;',len(task),'task checks;',len(noise),'noise conditions',flush=True)

if __name__=='__main__':
 with torch.inference_mode():main()
