"""Check the original HF hidden-state extraction against saved hook captures."""
from pathlib import Path
import importlib.util,json,torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pc_engine',ROOT/'scripts/run_positive_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=json.loads((ROOT/'configs/positive-control-v1.json').read_text());e=m.Engine(p['models'][0],p)
a=json.loads((ROOT/'reports/positive-control-v1/anchors.json').read_text())['anchors'][0]
ids=e.tokens(a['models']['qwen17']['token_ids']);saved=load_file(str(ROOT/'reports/positive-control-v1/qwen17/fixed-prefixes.safetensors'))['anchor_00'].to('mps')
report=[]
with torch.inference_mode():
 for cache in [False,True]:
  for mask in [False,True]:
   hooks={};handles=[]
   for i,layer in enumerate(e.model.model.layers):
    def save(module,args,out,i=i):hooks[i]=out.detach().clone()
    handles.append(layer.register_forward_hook(save))
   kwargs={'input_ids':ids,'output_hidden_states':True,'use_cache':cache}
   if mask:kwargs['attention_mask']=torch.ones_like(ids)
   out=e.model(**kwargs)
   for h in handles:h.remove()
   r={'use_cache':cache,'attention_mask':mask,'hidden_state_count':len(out.hidden_states),
      'saved_vs_hook2_max_abs':float((hooks[2]-saved).float().abs().max()),
      'hidden3_vs_hook2_max_abs':float((out.hidden_states[3]-hooks[2]).float().abs().max()),
      'hidden27_vs_hook26_max_abs':float((out.hidden_states[27]-hooks[26]).float().abs().max()),
      'last_token_saved_vs_hook2_max_abs':float((hooks[2][:,-1]-saved[:,-1]).float().abs().max()),
      'h3_norm':float(out.hidden_states[3][:,-1].float().norm())}
   report.append(r);print(r,flush=True)
(ROOT/'reports/implementation-audit-v1/capture-modes.json').write_text(json.dumps(report,indent=2)+'\n')
