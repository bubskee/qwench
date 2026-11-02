"""Bounded audit: replay three already selected pairs with legacy-sampler anchors.

Uses exported unmodified reference sweep/model/parametrization modules. Does not
alter the frozen v1 experiment or continue the 4B sweep.
"""
from pathlib import Path
import importlib.util,sys,os,json,pickle,time,hashlib
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/implementation-audit-v1'
REF=ROOT.parents[1]/'work/fsec-reference';RUNTIME=ROOT.parents[1]/'work/fsec-audit-runtime'
spec=importlib.util.spec_from_file_location('audit_engine',ROOT/'scripts/run_positive_control.py')
ours=importlib.util.module_from_spec(spec);spec.loader.exec_module(ours)
os.chdir(RUNTIME);sys.path.insert(0,str(RUNTIME))
from scripts.lib.sweep_core import run_2d_per_anchors
protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
engine=ours.Engine(protocol['models'][0],protocol)
engine.model._plateau_info={'model_type':'llama','d_model':engine.d_model}
hidden=load_file(str(OUT/'legacy-prefixes.safetensors'))
context=[hidden[f'anchor_{i:02d}'][:,:-1] for i in range(30)]
acts=torch.stack([hidden[f'anchor_{i:02d}'][0,-1].float() for i in range(30)])
directions=pickle.loads((REF/'results/directions/dirs_qwen_L2.pkl').read_bytes())['directions']
directions.update(pickle.loads((REF/'results/directions/dirs_qwen_L2_random.pkl').read_bytes())['directions'])
angles=np.asarray(protocol['angles_deg']);summary=[]
for pair in protocol['pairs']:
 path=OUT/('legacy_'+pair['id']+'.npz')
 if path.exists():raise SystemExit('Refusing to overwrite an audit result: '+str(path))
 start=time.perf_counter()
 result=run_2d_per_anchors(engine.model,context,acts,directions[pair['a']],directions[pair['b']],2,26,'mps',np.deg2rad(angles))
 np.savez_compressed(path,angles_deg=angles,l2=result['l2'])
 summary.append({'pair':pair,'seconds':time.perf_counter()-start,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'anchors_sha256':hashlib.sha256((OUT/'legacy-anchors.json').read_bytes()).hexdigest(),
  'directions':'Released original vectors; no refitting or selection',
  'runtime':'Native MPS BF16 SDPA, current torch/transformers, original unmodified suffix and sweep functions'})
 (OUT/'legacy-replay.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('COMPLETE',summary[-1],flush=True)
