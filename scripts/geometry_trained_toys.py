"""Prepare task-conditioned pairs, then measure frozen trained-toy geometry."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import importlib,os
T=importlib.import_module(os.environ.get('QWENCH_TRAINED_DRIVER','train_scalar_core'))
ROOT=T.ROOT;OUT=T.OUT;P=T.P;S=OUT/'surfaces';torch.set_num_threads(2)

def weights():
 assert (OUT/'TRAINING-COMPLETE.json').exists(),'Full training must finish before final geometry'
 data=load_file(str(OUT/'weights-100000.safetensors'));bank={}
 for arch in P['architectures']:
  for i,(exponent,seed) in enumerate(T.network_ids()):
   ident=f'{arch["name"]}_L{exponent}_s{seed}';roles=['gate','out']+(['value'] if arch['activation']=='swiglu' else [])
   bank[ident]={'activation':arch['activation'],**{role:data[f'{arch["name"]}.{role}'][i].double() for role in roles}}
 return bank

def forward_latent(z,model):
 n=model['gate'].shape[0];kind=model['activation'];g=z[...,:n]
 h=g.relu() if kind=='relu' else F.gelu(g,approximate='none') if kind=='gelu' else F.leaky_relu(g,negative_slope=P.get('leaky_slope',.1)) if kind=='leaky_relu' else g.relu().square() if kind=='relu2' else F.silu(g)
 if kind=='swiglu':h=h*z[...,n:]
 return h@model['out'].T

def encoder(model):return torch.cat([model['gate'],model['value']],dim=0) if model['activation']=='swiglu' else model['gate']

def prepare():
 assert not (OUT/'geometry-plan.json').exists(),'Do not overwrite sealed geometry selection'
 evaluation=json.loads((OUT/'eval-100000.json').read_text())['networks'];inputs=np.load(OUT/'geometry-inputs.npz');networks=[]
 for row in evaluation:
  competent=set(row['competent_feature_indices']);chosen=[]
  for pair in inputs['pair_order']:
   pair=[int(x) for x in pair]
   if all(j in competent for j in pair):chosen.append(pair)
   if len(chosen)==P['geometry']['eligible_pairs']:break
  pairs=[]
  for cohort,values in [('fixed',inputs['fixed_pairs'].tolist()),('competence_conditioned',chosen)]:
   for pair in values:
    old=next((p for p in pairs if sorted(p['features'])==sorted(pair)),None)
    if old:old['cohorts'].append(cohort)
    else:pairs.append({'features':pair,'cohorts':[cohort],'competent':all(j in competent for j in pair)})
  networks.append({'id':row['id'],'architecture':row['architecture'],'activation':row['activation'],'loss_exponent':row['loss_exponent'],'seed':row['seed'],
   'competent_features':len(competent),'conditioned_pairs':chosen,'pairs':pairs,'selection_shortfall':P['geometry']['eligible_pairs']-len(chosen)})
 plan={'networks':networks,'weights_sha256':T.sha(OUT/'weights-100000.safetensors'),'evaluation_sha256':T.sha(OUT/'eval-100000.json'),
  'input_sha256':T.sha(OUT/'geometry-inputs.npz'),'protocol_sha256':T.sha(ROOT/'configs'/(P['version']+'.json')),'source_sha256':T.sha(__file__),
  'selection':'Only frozen task competence is used. This file is written before measuring final-network p.'}
 T.write(OUT/'geometry-plan.json',plan);print('GEOMETRY_PREPARED',len(networks),'networks; shortfalls',[(r['id'],r['selection_shortfall']) for r in networks if r['selection_shortfall']],flush=True)

def pair_directions(enc,pair,seed):
 c=enc[:,pair].T.numpy();norms=np.linalg.norm(c,axis=1)
 if np.any(norms<=1e-12):return None,{'failure':'zero_encoder_column','norms':norms.tolist()}
 rho=float(c[0]@c[1]/np.prod(norms))
 if abs(rho)>=.999999:return None,{'failure':'collinear_encoder_columns','cosine':rho,'norms':norms.tolist()}
 rng=np.random.default_rng(seed);u=rng.normal(size=c.shape[1]);u/=np.linalg.norm(u);v=rng.normal(size=c.shape[1]);v-=v@u*u;v/=np.linalg.norm(v)
 random=np.stack([norms[0]*u,norms[1]*(rho*u+np.sqrt(1-rho*rho)*v)])
 assert np.abs(c@c.T-random@random.T).max()<1e-9
 return {'encoder_columns':c,'gram_matched_random':random},{'cosine':rho,'norms':norms.tolist(),'gram_error':float(np.abs(c@c.T-random@random.T).max())}

def grid(model,backgrounds,pair,value,directions,sign,points):
 axis=np.linspace(0,1.5,points);aa,bb=np.meshgrid(axis,axis,indexing='ij');coeff=torch.tensor(np.stack([sign[0]*aa.ravel(),sign[1]*bb.ravel()],axis=1),dtype=torch.float64)
 delta=coeff@torch.tensor(directions,dtype=torch.float64);x=torch.tensor(backgrounds.copy(),dtype=torch.float64);x[:,pair]=value
 z=x@encoder(model).T;clean=forward_latent(z,model)
 response=np.stack([(forward_latent(anchor[None]+delta,model)-ref).norm(dim=-1).numpy().reshape(points,points) for anchor,ref in zip(z,clean)])
 assert np.isfinite(response).all() and np.max(response[:,0,0])<1e-9
 return response

def sweep():
 plan=json.loads((OUT/'geometry-plan.json').read_text());assert plan['source_sha256']==T.sha(__file__);assert plan['weights_sha256']==T.sha(OUT/'weights-100000.safetensors')
 inputs=np.load(OUT/'geometry-inputs.npz');assert plan['input_sha256']==T.sha(OUT/'geometry-inputs.npz');bank=weights();S.mkdir(exist_ok=True)
 started=time.perf_counter()
 for index,network in enumerate(plan['networks']):
  ident=network['id'];path=S/(ident+'.npz');receipt=path.with_suffix('.json')
  if path.exists():
   prior=json.loads(receipt.read_text());assert prior['plan_sha256']==T.sha(OUT/'geometry-plan.json') and prior['raw_sha256']==T.sha(path);continue
  model=bank[ident];enc=encoder(model);rows=[];arrays=[];fine=[];finerows=[];direction_meta=[];case_start=time.perf_counter()
  first_eligible=network['conditioned_pairs'][0] if network['conditioned_pairs'] else None
  for pi,pair in enumerate(network['pairs']):
   directions,desc=pair_directions(enc,pair['features'],P['geometry']['selection_seed']+1000*index+pi);direction_meta.append({'features':pair['features'],**desc})
   if directions is None:continue
   for kind,d in directions.items():
    for value in P['geometry']['pair_anchor_values']:
     for sign in P['geometry']['signs']:
      arrays.append(grid(model,inputs['backgrounds'],pair['features'],value,d,sign,41));rows.append({'pair_index':pi,'features':pair['features'],'cohorts':pair['cohorts'],'competent':pair['competent'],'kind':kind,'anchor_value':value,'sign':sign})
    if first_eligible is not None and sorted(pair['features'])==sorted(first_eligible):
     fine.append(grid(model,inputs['backgrounds'],pair['features'],-.5,d,[1,1],81));finerows.append({'pair_index':pi,'features':pair['features'],'kind':kind,'anchor_value':-.5,'sign':[1,1]})
  np.savez_compressed(path,l2=np.stack(arrays) if arrays else np.empty((0,4,41,41)),fine_l2=np.stack(fine) if fine else np.empty((0,4,81,81)))
  T.write(receipt,{'network':network,'rows':rows,'fine_rows':finerows,'directions':direction_meta,'seconds':time.perf_counter()-case_start,
   'raw_sha256':T.sha(path),'plan_sha256':T.sha(OUT/'geometry-plan.json')})
  print('SURFACE',index+1,'/30',ident,len(rows),'grids; total seconds',round(time.perf_counter()-started,1),flush=True)
 T.write(OUT/'GEOMETRY-COMPLETE.json',{'networks':len(plan['networks']),'plan_sha256':T.sha(OUT/'geometry-plan.json'),'seconds':time.perf_counter()-started})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','sweep'],required=True);args=ap.parse_args()
 with torch.inference_mode():globals()[args.stage]()
