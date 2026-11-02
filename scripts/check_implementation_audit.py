"""Check audit provenance, complete grids, and preservation of frozen v1 data."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/implementation-audit-v1';PRIOR=ROOT/'reports/positive-control-v1';RUNTIME=ROOT.parents[1]/'work/fsec-audit-runtime'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
execution=json.loads((OUT/'execution-audit.json').read_text())
assert len(execution['hidden_state_checks'])==30
for row in execution['hidden_state_checks']:
 assert row['original_full_forward_h3_vs_saved_max_abs']==0
 assert row['default_cached_hidden_states27_vs_uncached_block26_max_abs']==0
assert all(row['max_absolute_error']==0 for row in execution['source_suffix_checks'])
export=json.loads((OUT/'reference-export.json').read_text())
for path,digest in export['exported_sources'].items():assert sha(RUNTIME/path)==digest
anchors=json.loads((OUT/'legacy-anchors.json').read_text());assert len(anchors['anchors'])==30
assert anchors['packages']['datasets']=='4.8.5'
for a in anchors['anchors']:
 assert hashlib.sha256(a['text'].encode()).hexdigest()==a['text_sha256']
 assert len(a['models']['qwen17']['token_ids'])==5
replays=json.loads((OUT/'legacy-replay.json').read_text());assert len(replays)==3
cases=0
for row in replays:
 path=OUT/('legacy_'+row['pair']['id']+'.npz');assert sha(path)==row['sha256']
 assert row['anchors_sha256']==sha(OUT/'legacy-anchors.json')
 data=np.load(path);assert data['l2'].shape==(30,31,31)
 assert np.array_equal(data['angles_deg'],np.arange(0,61,2))
 assert np.isfinite(data['l2']).all() and (data['l2']>=0).all()
 cases+=data['l2'].size
# The old report's prose may acquire a correction, but every frozen
# computational artifact must still match its pre-audit manifest.
old=json.loads(subprocess.check_output(['git','show','2fdc1fc:reports/positive-control-v1/manifest.json'],cwd=ROOT))
checked=0
for name,digest in old['files'].items():
 if Path(name).suffix in {'.npz','.safetensors','.json','.py'}:
  assert sha(PRIOR/name)==digest,name
  checked+=1
result={'all_assertions_passed':True,'legacy_surfaces':len(replays),'legacy_cases':cases,
 'source_function_AA_cases':2*30*31*31,'total_audit_cases':cases+2*30*31*31,
 'unchanged_v1_computational_artifacts':checked,'no_new_4b_inference':True,
 'historical_anchor_identity':'Strong per-anchor response fingerprint match; original token file unavailable.'}
(OUT/'quality-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
