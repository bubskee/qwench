"""Assemble completed, separately tracked scalar training cohorts without retraining."""
from pathlib import Path
import json,shutil
from safetensors.torch import load_file,save_file
import scalar_comparison_runtime as R
import train_scalar_core as C
import train_scalar_additions as E

def main():
 assert not (R.OUT/'TRAINING-COMPLETE.json').exists(),'Do not overwrite completed aggregation'
 assert C.sha(C.WORK/'data.safetensors')==E.sha(E.WORK/'data.safetensors')
 weights={};proof={};seals=[]
 for source in [C,E]:
  marker=source.OUT/'TRAINING-COMPLETE.json';done=json.loads(marker.read_text());assert done['step']==100000
  seal=json.loads((source.OUT/'input-seal.json').read_text());seals.append(seal);path=source.OUT/'weights-100000.safetensors'
  assert source.sha(path)==done['final_weights_sha256']
  loaded=load_file(str(path));assert not(set(weights)&set(loaded));weights.update(loaded)
  proof[source.P['version']]={'completion_sha256':source.sha(marker),'input_seal_sha256':source.sha(source.OUT/'input-seal.json'),'final_weights_sha256':source.sha(path),'source_sha256':source.sha(source.__file__)}
 for step in range(10000,100001,10000):
  rows=[];sources={}
  for source in [C,E]:
   path=source.OUT/f'eval-{step:06d}.json';rows+=json.loads(path.read_text())['networks'];sources[str(path.relative_to(R.ROOT))]=R.sha(path)
  assert len(rows)==30 and len({r['id'] for r in rows})==30
  R.write(R.OUT/f'eval-{step:06d}.json',{'step':step,'networks':rows,'source_evaluations':sources,'meaning':'Exact saved evaluations merged; no new inference.'})
 save_file(weights,str(R.OUT/'weights-100000.safetensors'))
 for name in ['geometry-inputs.npz','geometry-inputs.json']:shutil.copyfile(C.OUT/name,R.OUT/name)
 R.write(R.OUT/'assembly.json',{'source_cohorts':proof,'dataset_sha256':seals[0]['dataset_sha256'],'protocol_sha256':R.sha(R.ROOT/'configs/toy-scalar-comparison-v1.json'),'networks':30,'source_sha256':R.sha(__file__)})
 R.write(R.OUT/'TRAINING-COMPLETE.json',{'networks':30,'step':100000,'source_cohorts':proof,'final_weights_sha256':R.sha(R.OUT/'weights-100000.safetensors'),'meaning':'Logical aggregate of two independently validated source training cohorts.'})
 print('ASSEMBLED',30,'fully trained scalar networks',flush=True)

if __name__=='__main__':main()
