"""Independent source parity, geometric invariants, calibration and artifact checks."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
import run_toy_cis as R
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/toy-cis-v1'
prepared,inputs=R.load_inputs();bank=R.models();P=R.P
spec=importlib.util.spec_from_file_location('upstream_cis',R.REF/'small_models.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
checks=[];x=R.sparse(64,P['seed']+55)
for name,(wi,wo) in bank.items():
 if name in P['checkpoints']:
  source=upstream.SimpleMLP(100,50).double();source.load_state_dict({'W_in':wi,'W_out':wo})
  error=float((source(x)-R.forward(x,wi,wo)).detach().abs().max());assert error<1e-12
 else:error=0
 equivalence=[];random_errors=[]
 for row in prepared['pairs']:
  pair=row['features'];c=inputs[f'{name}_{row["id"]}_codeword'];r=inputs[f'{name}_{row["id"]}_matched_random']
  random_errors.append(float(np.abs(c@c.T-r@r.T).max()));assert random_errors[-1]<1e-12
  changed=x.clone();changed[:,pair[0]]+=.31;changed[:,pair[1]]-=.27
  lhs=R.forward(changed,wi,wo);rhs=torch.relu(x@wi.T+.31*wi[:,pair[0]]-.27*wi[:,pair[1]])@wo.T
  equivalence.append(float((lhs-rhs).abs().max()));assert equivalence[-1]<1e-12
 checks.append({'model':name,'upstream_forward_max_abs':error,'feature_input_vs_column_intervention_max_abs':max(equivalence),'random_vs_codeword_gram_max_abs':max(random_errors)})
meta=json.loads((OUT/'surfaces.json').read_text());data=np.load(OUT/'surfaces.npz');a=json.loads((OUT/'analysis.json').read_text())
assert data['l2'].shape==(432,4,41,41);assert data['post_l2'].shape==(144,1,41,41);assert data['fine_l2'].shape==(6,4,81,81)
assert len(a['records'])==432 and len(a['post_relu'])==144 and len(a['fine_checks'])==6
assert meta['raw_sha256']==hashlib.sha256((OUT/'surfaces.npz').read_bytes()).hexdigest()
for key in ['l2','post_l2','fine_l2']:
 assert np.isfinite(data[key]).all() and (data[key]>=0).all()
 assert np.max(data[key][...,0,0])<1e-12
assert ((data['gate_switch_fraction']>=0)&(data['gate_switch_fraction']<=1)).all()
# Check every linear-decoder control against the exact quadratic Gram formula.
axis=np.linspace(0,1.5,41);aa,bb=np.meshgrid(axis,axis,indexing='ij');linear_error=0.
for i,row in enumerate(meta['post_rows']):
 wi,wo=bank[row['model']];directions=inputs[f'{row["model"]}_{row["pair"]}_{row["kind"]}'];u,v=directions@wo.numpy().T
 q=aa*aa*(u@u)+bb*bb*(v@v)+2*row['sign'][0]*row['sign'][1]*aa*bb*(u@v)
 expected=np.sqrt(np.maximum(q,0));error=float(np.abs(data['post_l2'][i,0]-expected).max());linear_error=max(linear_error,error)
assert linear_error<1e-10,linear_error
# Calibration checks do not require a positive result from any learned model.
for row in a['analytic_controls']:
 if row['name']=='orthogonal_linear_p2':assert abs(row['p']-2)<.03
 if row['name']=='separable_square_p4':assert abs(row['p']-4)<.05
 if row['name']=='tilted_linear':assert row['quadratic']['mean_radial_frac']<.001
noise=json.loads((OUT/'corruption.json').read_text())['records'];assert len(noise)==45
for name in bank:
 baseline=[r for r in noise if r['model']==name and r['level']==0]
 for key in ['positive_active_mse','inactive_rmse','inactive_false_positive_rate_above_0p1']:
  assert max(r[key] for r in baseline)-min(r[key] for r in baseline)==0
result={'all_assertions_passed':True,'source_and_intervention_checks':checks,'linear_control_exact_quadratic_max_abs':linear_error,
 'main_surfaces':432,'flat_primary_fits':sum(r['fits']['0.5']['p'] is None for r in a['records']),
 'post_controls':144,'fine_checks':6,'inference_points':sum(data[k].size for k in ['l2','post_l2','fine_l2']),
 'fixed_noise_conditions':45,'fine_max_abs_delta_p':max(abs(r['delta_p']) for r in a['fine_checks']),
 'protocol_sha256':hashlib.sha256((ROOT/'configs/toy-cis-v1.json').read_bytes()).hexdigest()}
(OUT/'quality-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
