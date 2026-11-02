"""Continue 18 scalar-activation toy networks under the frozen matched-activation protocol."""
from pathlib import Path
import argparse,hashlib,json,time,subprocess,os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/toy-scalars-core-v1'
WORK=ROOT.parents[1]/'work/toy-scalars-core-v1';P=json.loads((ROOT/'configs/toy-scalars-core-v1.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def initialize(device):
 base={}
 for seed in P['seeds']:
  g=torch.Generator().manual_seed(seed)
  base[seed]=(torch.rand(50,100,generator=g)*.2-.1,torch.rand(100,50,generator=g)*.3-.15,torch.rand(50,100,generator=g)*.2-.1)
 groups={};params=[]
 for arch in P['architectures']:
  width=arch['width'];members={}
  for role,index in [('gate',0),('out',1)]+([('value',2)] if arch['activation']=='swiglu' else []):
   copies=[]
   for exponent in P['loss_exponents']:
    for seed in P['seeds']:
     v=base[seed][index];copies.append((v[:width,:] if role!='out' else v[:,:width]).clone())
   members[role]=torch.stack(copies).to(device).contiguous().requires_grad_();params.append(members[role])
  groups[arch['name']]=members
 return groups,params

def hidden(x,g,activation):
 z=x@g['gate'].transpose(-1,-2)
 if activation=='relu':return z.relu()
 if activation=='gelu':return F.gelu(z,approximate='none')
 if activation=='silu':return F.silu(z)
 if activation=='swiglu':return F.silu(z)*(x@g['value'].transpose(-1,-2))
 raise ValueError(activation)
def predict(x,g,activation):return hidden(x,g,activation)@g['out'].transpose(-1,-2)
def network_ids():return [(e,s) for e in P['loss_exponents'] for s in P['seeds']]
def pool():
 path=WORK/'data.safetensors'
 if not path.exists():
  arrays={}
  for name,count,seed in [('train',P['task']['train_pool_size'],P['task']['train_data_seed']),('eval',P['task']['eval_size'],P['task']['eval_seed'])]:
   gen=torch.Generator().manual_seed(seed);mask=torch.rand(count,100,generator=gen)<.02;values=torch.rand(count,100,generator=gen)*2-1;arrays[name]=values*mask
  save_file(arrays,str(path))
 return path,load_file(str(path))

@torch.inference_mode()
def evaluate(groups,x):
 rows=[];counts=(x>0).sum(0);n_inactive=int((x==0).sum());n_negative=int((x<0).sum());eye=torch.eye(100,device=x.device)
 for arch in P['architectures']:
  g=groups[arch['name']];errsum=torch.zeros(6,100,device=x.device);inactive=torch.zeros(6,device=x.device);negative=inactive.clone();fp=inactive.clone();isolated=inactive.clone();multi=inactive.clone();nc_iso=0;nc_multi=0
  for start in range(0,len(x),4096):
   xx=x[start:start+4096];target=xx.relu();pred=predict(xx,g,arch['activation']);err=(pred-target).square();pos=xx>0;off=xx==0;neg=xx<0
   errsum+=(err*pos).sum(1);inactive+=(err*off).sum((1,2));negative+=(err*neg).sum((1,2));fp+=((pred>.1)*off).sum((1,2))
   count=(xx!=0).sum(-1);iso=pos&(count==1)[:,None];mul=pos&(count>=2)[:,None]
   isolated+=(err*iso).sum((1,2));multi+=(err*mul).sum((1,2));nc_iso+=int(iso.sum());nc_multi+=int(mul.sum())
  per=errsum/counts.clamp_min(1);h=hidden(eye,g,arch['activation']);single=h@g['out'].transpose(-1,-2);gains=single.diagonal(dim1=1,dim2=2)
  arg=single.argmax(-1)==torch.arange(100,device=x.device)[None];eligible=(per<.05)&(gains>=.25)&arg
  contribution=h*g['out'];absq=contribution.abs();feature_pr=absq.sum(-1).square()/absq.square().sum(-1).clamp_min(1e-30)
  supported=absq*eligible[:,:,None];neuron_pr=supported.sum(1).square()/supported.square().sum(1).clamp_min(1e-30)
  tensors={'per_feature_mse':per,'single_gains':gains,'eligible':eligible,'feature_pr':feature_pr,'neuron_pr':neuron_pr,'inactive':(inactive/n_inactive).sqrt(),'negative':(negative/n_negative).sqrt(),'fp':fp/n_inactive,'isolated':isolated/max(nc_iso,1),'multi':multi/max(nc_multi,1)}
  cpu={k:v.cpu().numpy() for k,v in tensors.items()}
  for i,(exponent,seed) in enumerate(network_ids()):
   mask=cpu['eligible'][i];n=int(mask.sum());usage=cpu['neuron_pr'][i];usage=usage[usage>0]
   rows.append({'id':f'{arch["name"]}_L{exponent}_s{seed}','architecture':arch['name'],'activation':arch['activation'],'width':arch['width'],'parameters':arch['parameters'],'loss_exponent':exponent,'seed':seed,
    'competent_features':n,'competent_feature_indices':np.where(mask)[0].tolist(),'per_feature_positive_mse':cpu['per_feature_mse'][i].tolist(),'single_feature_gains':cpu['single_gains'][i].tolist(),
    'mean_per_feature_positive_mse':float(cpu['per_feature_mse'][i].mean()),'inactive_rmse':float(cpu['inactive'][i]),'inactive_false_positive_rate':float(cpu['fp'][i]),'negative_rmse':float(cpu['negative'][i]),
    'isolated_active_mse':float(cpu['isolated'][i]),'multi_input_active_mse':float(cpu['multi'][i]),
    'feature_contribution_participation':cpu['feature_pr'][i].tolist(),'neuron_feature_participation':cpu['neuron_pr'][i].tolist(),
    'mean_competent_feature_participation':float(cpu['feature_pr'][i][mask].mean()) if n else None,'mean_neuron_feature_participation':float(usage.mean()) if len(usage) else None})
 return rows

def main(resume=False):
 OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);torch.set_num_threads(P['training']['cpu_threads']);assert torch.backends.mps.is_available()
 device='mps';groups,params=initialize(device);opt=torch.optim.Adam(params,lr=P['training']['lr'],betas=tuple(P['training']['betas']),eps=P['training']['eps'],weight_decay=0,foreach=True)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,P['training']['steps']);data_path,data=pool();x=data['train'].to(device);ev=data['eval'].to(device);del data
 deps={'source_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'configs/toy-scalars-core-v1.json'),'dataset_sha256':sha(data_path)}
 checkpoint=WORK/'resume.pt';startstep=0;elapsed_prior=0.
 if resume:
  saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['dependencies']==deps
  for name,g in groups.items():
   for role,w in g.items():w.data.copy_(saved['weights'][name][role].to(device))
  opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);torch.mps.set_rng_state(saved['rng']);startstep=saved['step'];elapsed_prior=saved['seconds']
 else:
  assert not checkpoint.exists() and not (OUT/'training.jsonl').exists(),'Use --resume or a new version; do not overwrite a training run'
  torch.mps.manual_seed(P['training']['minibatch_seed'])
  write(OUT/'input-seal.json',{**deps,'protocol_commit':'ae009ef','git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'device':device,'dataset_path':str(data_path),'network_ids':[f'{a["name"]}_L{e}_s{s}' for a in P['architectures'] for e,s in network_ids()]})
 exponents=torch.tensor([e for e,s in network_ids()],device=device)[:,None,None]
 wall=time.perf_counter();last_sync=wall
 for step in range(startstep,P['training']['steps']):
  idx=torch.randint(len(x),(P['training']['batch_size'],),device=device);batch=x[idx];target=batch.relu();opt.zero_grad(set_to_none=True);losses=[]
  for arch in P['architectures']:
   prediction=predict(batch,groups[arch['name']],arch['activation']);losses.append((prediction-target).abs().pow(exponents).mean((1,2)))
  loss=torch.stack(losses).sum();loss.backward();opt.step();sched.step()
  completed=step+1
  if completed%P['training']['log_every']==0 or completed==1:
   vals=torch.stack(losses).detach().cpu().numpy();assert np.isfinite(vals).all(),'Nonfinite training: stop without silently tuning the protocol'
   now=time.perf_counter();row={'step':completed,'total_steps':P['training']['steps'],'seconds':elapsed_prior+now-wall,'lr':sched.get_last_lr()[0],
    'loss_by_architecture':{a['name']:vals[i].tolist() for i,a in enumerate(P['architectures'])}}
   with (OUT/'training.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
   write(OUT/'progress.json',row);print('TRAIN',completed,'seconds',round(row['seconds'],1),'losses',{a['name']:[round(float(v),7) for v in vals[i]] for i,a in enumerate(P['architectures'])},flush=True)
  if completed%P['training']['validation_every']==0:
   validation=evaluate(groups,ev);write(OUT/f'eval-{completed:06d}.json',{'step':completed,'networks':validation,'dependencies':deps})
   print('EVAL',completed,{a['name']:{f'L{e}':[r['competent_features'] for r in validation if r['architecture']==a['name'] and r['loss_exponent']==e] for e in P['loss_exponents']} for a in P['architectures']},flush=True)
  if completed%P['training']['checkpoint_every']==0:
   weights={n:{r:w.detach().cpu() for r,w in g.items()} for n,g in groups.items()}
   save_file({f'{n}.{r}':w for n,g in weights.items() for r,w in g.items()},str(OUT/f'weights-{completed:06d}.safetensors'))
   saved={'step':completed,'seconds':elapsed_prior+time.perf_counter()-wall,'weights':weights,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':torch.mps.get_rng_state(),'dependencies':deps}
   temp=checkpoint.with_suffix('.tmp');torch.save(saved,temp);temp.replace(checkpoint)
 write(OUT/'TRAINING-COMPLETE.json',{'step':P['training']['steps'],'networks':len(P['architectures'])*len(network_ids()),'seconds':elapsed_prior+time.perf_counter()-wall,'dependencies':deps,'final_weights_sha256':sha(OUT/'weights-100000.safetensors')})
 print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();main(args.resume)
