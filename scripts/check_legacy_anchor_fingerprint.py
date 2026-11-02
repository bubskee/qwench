"""Check recovered inputs against the released mean-norm fingerprint, before p."""
from pathlib import Path
import hashlib,json
import torch
from safetensors.torch import save_file
from run_positive_control import Engine
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/implementation-audit-v1'
protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
anchors=json.loads((OUT/'legacy-anchors.json').read_text())['anchors']
prior=json.loads((ROOT/'reports/positive-control-v1/anchors.json').read_text())['anchors']
engine=Engine(protocol['models'][0],protocol)
hidden={}
for i,a in enumerate(anchors):
 h=engine.capture(engine.tokens(a['models']['qwen17']['token_ids']),2,all_tokens=True).cpu()
 hidden[f'anchor_{i:02d}']=h
norms=[float(h[0,-1].float().norm()) for h in hidden.values()]
mean=float(torch.tensor(norms).mean())
reference=json.loads((ROOT/'reports/positive-control-v1/analysis/reference_gender_age.json').read_text())['anchor_norm_mean']
result={'norms':norms,'mean_anchor_norm':mean,'released_mean_anchor_norm':reference,'absolute_difference':abs(mean-reference),
 'prior_mean_anchor_norm':json.loads((ROOT/'reports/positive-control-v1/qwen17/prepared.json').read_text())['mean_anchor_norm'],
 'same_source_documents_vs_prior':len({a['source']['id'] for a in anchors}&{a['source']['id'] for a in prior}),
 'same_token_windows_vs_prior':len({tuple(a['models']['qwen17']['token_ids']) for a in anchors}&{tuple(a['models']['qwen17']['token_ids']) for a in prior}),
 'anchors_sha256':hashlib.sha256((OUT/'legacy-anchors.json').read_bytes()).hexdigest(),
 'interpretation':'A mean-norm fingerprint can support identity but cannot establish historical token identity without the original token file.'}
save_file(hidden,str(OUT/'legacy-prefixes.safetensors'))
(OUT/'legacy-fingerprint.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
