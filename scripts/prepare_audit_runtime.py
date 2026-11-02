"""Export the exact reference modules used by the implementation A/A audit."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];REF=ROOT.parents[1]/'work/fsec-reference';OUT=ROOT.parents[1]/'work/fsec-audit-runtime'
revision=json.loads((ROOT/'configs/positive-control-v1.json').read_text())['reference_revision']
paths=['src/model.py','scripts/lib/sweep_core.py','scripts/lib/parametrize.py','scripts/lib/superellipse.py','scripts/lib/activations.py']
records={}
for name in paths:
 value=subprocess.check_output(['git','show',revision+':'+name],cwd=REF)
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
 records[name]=hashlib.sha256(value).hexdigest()
for name in ['src/__init__.py','scripts/__init__.py','scripts/lib/__init__.py']:(OUT/name).write_text('')
print(json.dumps({'reference_revision':revision,'exported_sources':records},indent=2))
