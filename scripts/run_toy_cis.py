"""Frozen CPU pilot: released CiS checkpoints, exact controls, geometry and noise.

No training and no remote mutations. See configs/toy-cis-v1.json.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,time,importlib.metadata
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/toy-cis-v1'
REF=ROOT.parents[1]/'work/toy-model-cis-reference'
P=json.loads((ROOT/'configs/toy-cis-v1.json').read_text())
torch.set_num_threads(6)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
def models():
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REF,text=True).strip()
 assert revision==P['reference_revision']
 result={}
 for name,path in P['checkpoints'].items():
  weights=torch.load(REF/path,map_location='cpu',weights_only=True)
  result[name]=(weights['W_in'].double(),weights['W_out'].double())
 result['dedicated']=(torch.eye(100,dtype=torch.float64),torch.eye(100,dtype=torch.float64))
 return result

def sparse(count,seed):
 rng=np.random.default_rng(seed)
 return tensor((rng.random((count,100))<P['activation_probability'])*rng.uniform(-1,1,(count,100)))
def forward(x,wi,wo):return torch.relu(x@wi.T)@wo.T

def prepare():
 assert not (OUT/'prepared.json').exists(),'Refusing to replace prepared inputs'
 bank=models();x=sparse(P['evaluation_samples'],P['seed']);target=x.relu()
 metadata={};eligible={};metrics={}
 for name,(wi,wo) in bank.items():
  single=wi.T.relu()@wo.T;gain=single.diag();norms=wi.norm(dim=0)
  accepted=(gain>=P['eligibility']['min_single_feature_self_gain'])&(single.argmax(-1)==torch.arange(100))&(norms>1e-8)
  eligible[name]=set(torch.where(accepted)[0].tolist())
  prediction=forward(x,wi,wo);positive=x>0;inactive=x==0;negative=x<0
  err=(prediction-target).square()
  per=(err*positive).sum(0)/positive.sum(0).clamp_min(1)
  pinv=torch.linalg.pinv(wi);cos=float((wo*pinv).sum()/(wo.norm()*pinv.norm()))
  metadata[name]={'encoder_shape':list(wi.shape),'eligible_features':sorted(eligible[name]),'eligible_count':int(accepted.sum()),
   'single_feature_self_gains':gain.tolist(),'single_feature_natural_argmax_correct':int((single.argmax(-1)==torch.arange(100)).sum()),
   'positive_active_mse':float(err[positive].mean()),'inactive_rmse':float(err[inactive].mean().sqrt()),'negative_input_rmse':float(err[negative].mean().sqrt()),
   'inactive_false_positive_rate_above_0p1':float((prediction[inactive]>.1).double().mean()),
   'per_feature_positive_active_mse':per.tolist(),'codeword_sizes_above_0p05':(wi>.05).sum(0).tolist(),
   'decoder_pseudoinverse_cosine':cos,'median_encoder_column_norm':float(norms.median()),
   'checkpoint_sha256':sha(REF/P['checkpoints'][name]) if name in P['checkpoints'] else None}
  print('MODEL',name,'eligible',int(accepted.sum()),'active MSE',float(err[positive].mean()),flush=True)
 common=sorted(set.intersection(*eligible.values()));assert len(common)>=2*P['pair_count']
 rng=np.random.default_rng(P['seed']);chosen=rng.permutation(common)[:2*P['pair_count']].reshape(-1,2).tolist()
 backgrounds=sparse(P['backgrounds'],P['seed']+1).numpy()
 arrays={'backgrounds':backgrounds};pairmeta=[]
 for pairindex,pair in enumerate(chosen):
  row={'id':pairindex,'features':pair,'models':{}}
  for modelindex,(name,(wi,wo)) in enumerate(bank.items()):
   c1,c2=wi[:,pair[0]].numpy(),wi[:,pair[1]].numpy();n1,n2=np.linalg.norm(c1),np.linalg.norm(c2)
   rho=float(c1@c2/(n1*n2));assert abs(rho)<.99999
   rg=np.random.default_rng(P['seed']+1000+100*modelindex+pairindex)
   u=rg.normal(size=len(c1));u/=np.linalg.norm(u);v=rg.normal(size=len(c1));v-=u@v*u;v/=np.linalg.norm(v)
   random=np.stack([n1*u,n2*(rho*u+np.sqrt(1-rho*rho)*v)])
   arrays[f'{name}_{pairindex}_codeword']=np.stack([c1,c2]);arrays[f'{name}_{pairindex}_matched_random']=random
   natural=np.stack([c1/n1,c2/n2]);gram=natural@natural.T
   support=wi[:,pair].numpy()>.05;union=np.logical_or(support[:,0],support[:,1]).sum()
   row['models'][name]={'cosine':rho,'norms':[float(n1),float(n2)],'support_jaccard':float(np.logical_and(support[:,0],support[:,1]).sum()/union) if union else None}
  pairmeta.append(row)
 np.savez_compressed(OUT/'inputs.npz',**arrays)
 write(OUT/'prepared.json',{'protocol_sha256':sha(ROOT/'configs/toy-cis-v1.json'),'source_sha256':sha(__file__),
  'source_revision':P['reference_revision'],'preparation_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'models':metadata,'pairs':pairmeta,'common_eligible_features':common,
  'input_sha256':sha(OUT/'inputs.npz'),'evaluation_input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
  'packages':{s:importlib.metadata.version(s) for s in ['torch','numpy']},
  'scope':'Selection and behavior only; no contour exponents or corruption outcomes measured.'})
 print('PAIRS',chosen,flush=True)

def load_inputs():
 prepared=json.loads((OUT/'prepared.json').read_text())
 assert prepared['protocol_sha256']==sha(ROOT/'configs/toy-cis-v1.json')
 assert prepared['source_sha256']==sha(__file__),'Use the frozen source, or version the protocol'
 assert prepared['input_sha256']==sha(OUT/'inputs.npz')
 for name,path in P['checkpoints'].items():assert sha(REF/path)==prepared['models'][name]['checkpoint_sha256']
 return prepared,np.load(OUT/'inputs.npz')

def one_grid(wi,wo,backgrounds,pair,value,directions,sign,n,stage='pre'):
 amplitude=np.linspace(0,P['max_feature_amplitude'],n)
 a,b=np.meshgrid(amplitude,amplitude,indexing='ij')
 d=tensor(np.stack([sign[0]*a.ravel(),sign[1]*b.ravel()],axis=1))@tensor(directions)
 x=tensor(backgrounds.copy());x[:,pair]=value;z=x@wi.T;clean=z.relu()@wo.T;rows=[];switches=[]
 for anchor,ref in zip(z,clean):
  changed=(anchor[None]+d).relu() if stage=='pre' else anchor.relu()[None]+d
  out=changed@wo.T;rows.append((out-ref).norm(dim=-1).numpy().reshape(n,n))
  switches.append(((anchor[None]+d>0)!=(anchor>0)[None]).double().mean(-1).numpy().reshape(n,n))
 return np.stack(rows),np.stack(switches)

def geometry():
 assert not (OUT/'surfaces.npz').exists(),'Refusing to overwrite measured surfaces'
 prepared,inputs=load_inputs();bank=models();rows=[];grids=[];switches=[];postrows=[];postgrids=[];fine=[];finerows=[]
 start=time.perf_counter()
 for name,(wi,wo) in bank.items():
  for pair in prepared['pairs']:
   for kind in P['direction_kinds']:
    directions=inputs[f'{name}_{pair["id"]}_{kind}']
    for sign in P['signs']:
     for state,value in P['anchor_pair_values'].items():
      l2,flip=one_grid(wi,wo,inputs['backgrounds'],pair['features'],value,directions,sign,P['grid_points'])
      assert np.all(l2[:,0,0]<1e-12) and np.isfinite(l2).all()
      rows.append({'model':name,'pair':pair['id'],'kind':kind,'sign':sign,'state':state});grids.append(l2);switches.append(flip)
     l2,_=one_grid(wi,wo,inputs['backgrounds'][:1],pair['features'],0,directions,sign,P['grid_points'],'post')
     postrows.append({'model':name,'pair':pair['id'],'kind':kind,'sign':sign,'state':'post_relu'});postgrids.append(l2)
    if pair['id']==0:
     l2,_=one_grid(wi,wo,inputs['backgrounds'],pair['features'],0,directions,[1,1],81)
     finerows.append({'model':name,'pair':pair['id'],'kind':kind,'sign':[1,1],'state':'inactive'});fine.append(l2)
  print('GEOMETRY',name,'seconds',time.perf_counter()-start,flush=True)
 np.savez_compressed(OUT/'surfaces.npz',l2=np.stack(grids),gate_switch_fraction=np.stack(switches),post_l2=np.stack(postgrids),fine_l2=np.stack(fine))
 write(OUT/'surfaces.json',{'rows':rows,'post_rows':postrows,'fine_rows':finerows,'seconds':time.perf_counter()-start,'source_sha256':sha(__file__),'raw_sha256':sha(OUT/'surfaces.npz')})
 print('GEOMETRY_COMPLETE',len(rows),'surfaces',len(postrows),'post controls',flush=True)

def corruption():
 assert not (OUT/'corruption.json').exists(),'Refusing to overwrite corruption probe'
 prepared,inputs=load_inputs();x=sparse(P['corruption_samples'],P['seed']+2);target=x.relu()
 rng=np.random.default_rng(P['seed']+3);raw=tensor(rng.normal(size=(len(x),100)));records=[]
 for name,(wi,wo) in models().items():
  z=x@wi.T;base=z.relu();scale=prepared['models'][name]['median_encoder_column_norm']
  unit=raw[:,:len(wi)];unit=unit/unit.norm(dim=-1,keepdim=True)
  for level in P['corruption_levels']:
   delta=unit*(level*scale)
   methods={'natural_relu':(z+delta).relu(),'frozen_gates':base+(z>0)*delta,'decoder_only':base+delta}
   for method,hidden in methods.items():
    prediction=hidden@wo.T;err=(prediction-target).square();active=x>0;inactive=x==0;negative=x<0
    per=(err*active).sum(0)/active.sum(0).clamp_min(1)
    records.append({'model':name,'level':level,'method':method,'positive_active_mse':float(err[active].mean()),
     'inactive_rmse':float(err[inactive].mean().sqrt()),'negative_input_rmse':float(err[negative].mean().sqrt()),
     'inactive_false_positive_rate_above_0p1':float((prediction[inactive]>.1).double().mean()),
     'per_feature_positive_active_mse':per.tolist(),'error_on_eligible_positive_features':float(per[prepared['models'][name]['eligible_features']].mean())})
  print('CORRUPTION',name,flush=True)
 write(OUT/'corruption.json',{'records':records,'samples':len(x),'clean_input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
  'source_sha256':sha(__file__),'target':'ReLU(clean x) remains fixed at every noise level.'})

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['prepare','geometry','corruption'],required=True);args=ap.parse_args()
 OUT.mkdir(exist_ok=True);globals()[args.stage]()
