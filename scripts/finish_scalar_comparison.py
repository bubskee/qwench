"""Finish the authorized local experiment after both training cohorts complete."""
from pathlib import Path
import json,os,subprocess,time,traceback
import scalar_comparison_runtime as R
import train_scalar_core as C
import train_scalar_additions as E
PYTHON='/Users/bubskee/workspace/qwen35-4b-lenses/.venv/bin/python'
UV='/Users/bubskee/.local/bin/uv'

def main():
 deadline=time.monotonic()+4*3600
 while not all((m.OUT/'TRAINING-COMPLETE.json').exists() for m in [C,E]):
  if time.monotonic()>deadline:raise RuntimeError('Training completion wait exceeded four hours')
  time.sleep(5)
 env=os.environ.copy();env['QWENCH_TRAINED_DRIVER']='scalar_comparison_runtime'
 def run(stage,args,plot=False):
  R.write(R.OUT/'pipeline-state.json',{'stage':stage,'command':args})
  print('STAGE',stage,flush=True)
  active=env.copy()
  if plot:active['PYTHONPATH']='/Users/bubskee/workspace/qwen35-4b-lenses/.venv/lib/python3.12/site-packages'
  subprocess.run(args,cwd=R.ROOT,env=active,check=True)
 def analysis(script):return [UV,'run','--no-project','--python',PYTHON,'--with-requirements','requirements-reproduction.txt','python',script]
 run('assemble',[PYTHON,'scripts/assemble_scalar_comparison.py'])
 run('implementation-check',[PYTHON,'scripts/check_trained_geometry.py'])
 run('prepare-geometry',[PYTHON,'scripts/geometry_trained_toys.py','--stage','prepare'])
 names=['assembly.json','TRAINING-COMPLETE.json','weights-100000.safetensors','eval-100000.json','geometry-plan.json','geometry-inputs.npz','geometry-inputs.json','implementation-checks-100000.json']
 paths=[str((R.OUT/name).relative_to(R.ROOT)) for name in names]
 run('seal-pairs',['git','add',*paths])
 run('commit-pairs',['git','commit','-m','Seal final trained scalar weights, task checks and pair selection before p fits','--only',*paths])
 run('geometry',[PYTHON,'scripts/geometry_trained_toys.py','--stage','sweep'])
 run('fit',analysis('scripts/fit_trained_toys.py'),plot=True)
 run('plots',analysis('scripts/plot_trained_toys.py'),plot=True)
 run('summaries',[PYTHON,'scripts/summarize_scalar_comparison.py'])
 run('audit',[PYTHON,'scripts/audit_trained_toys.py'])
 R.write(R.OUT/'PIPELINE-COMPLETE.json',{'data_stages_complete':True,'remaining':'Human-readable findings note and visual inspection; no remote publication.'})
 print('DATA_PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  R.write(R.OUT/'PIPELINE-FAILED.json',{'error':repr(error)});traceback.print_exc();raise
