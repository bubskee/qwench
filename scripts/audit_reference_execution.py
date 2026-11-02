"""Focused A/A replay using exported, unmodified reference sweep functions."""
from pathlib import Path
import importlib.util,os,sys,json,time,hashlib
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
REFRUN=ROOT.parents[1]/'work/fsec-audit-runtime'
OUT=ROOT/'reports/implementation-audit-v1';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('qwench_positive_engine',ROOT/'scripts/run_positive_control.py')
ours=importlib.util.module_from_spec(spec);spec.loader.exec_module(ours)
os.chdir(REFRUN);sys.path.insert(0,str(REFRUN))
from src import model as source_model
from scripts.lib import sweep_core as source_sweep
from scripts.lib import parametrize as source_param

protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text());pin=protocol['models'][0]
prior=ROOT/'reports/positive-control-v1/qwen17'
anchors=json.loads((ROOT/'reports/positive-control-v1/anchors.json').read_text())['anchors']
prefixes=load_file(str(prior/'fixed-prefixes.safetensors'))
directions=load_file(str(prior/'directions.safetensors'))
engine=ours.Engine(pin,protocol)
engine.model._plateau_info={'model_type':'llama','d_model':engine.d_model}
for i,row in enumerate(anchors):engine.install_anchor(i,row['models']['qwen17']['token_ids'],prefixes[f'anchor_{i:02d}'])
angles=np.asarray(protocol['angles_deg']);radians=np.deg2rad(angles)

checks=[]
with torch.inference_mode():
 for i,row in enumerate(anchors):
  ids=engine.tokens(row['models']['qwen17']['token_ids'])
  result=engine.model(ids,output_hidden_states=True)
  saved=prefixes[f'anchor_{i:02d}'].to('mps')
  error=float((result.hidden_states[3]-saved).float().abs().max())
  measured=engine.capture(ids,26)
  measured_error=float((result.hidden_states[27][:,-1]-measured).float().abs().max())
  check={'anchor':i,'original_full_forward_h3_vs_saved_max_abs':error,'default_cached_hidden_states27_vs_uncached_block26_max_abs':measured_error}
  print('CAPTURE',check,flush=True);checks.append(check)
  (OUT/'capture-checks-partial.json').write_text(json.dumps(checks,indent=2)+'\n')
  assert error==0
 coords=[(float(a),float(b)) for a in angles for b in angles]
 d1=directions['Gender'].to('mps');d2=directions['Age'].to('mps')
 changed,error=engine.changed_hidden(0,len(coords),d1,d2,coords)
 a=prefixes['anchor_00'][0,-1].float().to('mps')
 u,v=source_param.orthonormalize(a/a.norm(),d1,d2)
 points=source_sweep.build_perturbed_pts_2d(a,u,v,torch.tensor(radians,dtype=torch.float32,device='mps')).to(torch.bfloat16)
 delta=changed[:,-1].float()-points.float()
 point_check={'max_absolute_bf16_component_difference':float(delta.abs().max()),
              'mean_absolute_bf16_component_difference':float(delta.abs().mean()),
              'different_components':int((delta!=0).sum()),'total_components':delta.numel(),
              'max_relative_point_l2_difference':float(delta.norm(dim=-1).max()/a.norm())}
 suffix=[]
 for count in [1,8,64]:
  part=changed[:count].contiguous()
  native=engine.replay(0,count,part)
  source=source_model.forward_from_layer_to_layer(engine.model,part[:1,:-1],part[:,-1:],2,26).float()
  error=float((native-source).abs().max());suffix.append({'batch':count,'max_absolute_error':error});assert error==0
print('COMPONENT_AUDIT',json.dumps({'point_check':point_check,'suffix':suffix}),flush=True)
summary={'scope':'Implementation audit only: same stored anchors/checkpoint; not a new feature survey.',
         'reference_sources':{str(p.relative_to(REFRUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in REFRUN.rglob('*.py')},
         'hidden_state_checks':checks,'point_check':point_check,'source_suffix_checks':suffix,
         'model_attention_implementation':engine.model.config._attn_implementation,'cases':[]}
if os.environ.get('QWENCH_AUDIT_CHECKS_ONLY'):
 (OUT/'execution-checks-corrected.json').write_text(json.dumps(summary,indent=2)+'\n')
 raise SystemExit(0)
context=[prefixes[f'anchor_{i:02d}'][:,:-1] for i in range(30)]
acts=torch.stack([prefixes[f'anchor_{i:02d}'][0,-1].float() for i in range(30)])
import pickle
original=pickle.loads((ROOT.parents[1]/'work/fsec-reference/results/directions/dirs_qwen_L2.pkl').read_bytes())['directions']
for label,bank in [('fresh_directions',directions),('released_directions',original)]:
 start=time.perf_counter()
 result=source_sweep.run_2d_per_anchors(engine.model,context,acts,bank['Gender'],bank['Age'],2,26,'mps',radians)
 np.savez_compressed(OUT/(label+'.npz'),angles_deg=angles,l2=result['l2'])
 baseline=np.load(prior/'gender_age.npz')['l2_paper']
 diff=np.abs(result['l2']-baseline)
 summary['cases'].append({'case':label,'seconds':time.perf_counter()-start,
   'max_absolute_l2_difference_vs_saved':float(diff.max()),'mean_absolute_l2_difference_vs_saved':float(diff.mean()),
   'mean_relative_l2_difference_vs_saved':float(diff.mean()/baseline.mean())})
 (OUT/'execution-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('CASE_COMPLETE',summary['cases'][-1],flush=True)
print('COMPLETE',flush=True)
