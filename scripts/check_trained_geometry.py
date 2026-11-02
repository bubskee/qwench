"""Check gated intervention semantics and trained-weight provenance without using p."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import train_toy_activations as T
import geometry_trained_toys as G

def main(step):
 torch.set_num_threads(2);raw=load_file(str(T.OUT/f'weights-{step:06d}.safetensors'));g=torch.Generator().manual_seed(77123)
 x=(torch.rand(64,100,generator=g,dtype=torch.float64)*2-1)*(torch.rand(64,100,generator=g)<.02)
 inputs=np.load(T.OUT/'geometry-inputs.npz');checks=[]
 for arch in T.P['architectures']:
  for index,(exp,seed) in enumerate(T.network_ids()):
   model={'activation':arch['activation'],**{role:raw[f'{arch["name"]}.{role}'][index].double() for role in ['gate','out']+(['value'] if arch['activation']=='swiglu' else [])}}
   enc=G.encoder(model);latent=x@enc.T;expected=T.predict(x,model,arch['activation']);actual=G.forward_latent(latent,model)
   parity=float((expected-actual).abs().max());assert parity<1e-10,parity
   differences=[];grams=[]
   for pi,pair in enumerate(inputs['fixed_pairs'].tolist()):
    changed=x.clone();changed[:,pair[0]]+=.37;changed[:,pair[1]]-=.22
    input_output=T.predict(changed,model,arch['activation']);hidden_output=G.forward_latent(latent+.37*enc[:,pair[0]]-.22*enc[:,pair[1]],model)
    error=float((input_output-hidden_output).abs().max());differences.append(error);assert error<1e-10,error
    directions,desc=G.pair_directions(enc,pair,88123+pi)
    if directions is not None:grams.append(desc['gram_error'])
   checks.append({'id':f'{arch["name"]}_L{exp}_s{seed}','latent_dimensions':len(enc),'forward_parity_max_abs':parity,'column_vs_input_intervention_max_abs':max(differences),'gram_matching_max_abs':max(grams) if grams else None})
 result={'all_assertions_passed':True,'checkpoint_step':step,'checkpoint_sha256':T.sha(T.OUT/f'weights-{step:06d}.safetensors'),'networks':checks,
  'meaning':'Full SwiGLU perturbations address both independently learned gate and value preactivations. No p outcome is used by these checks.'}
 T.write(T.OUT/f'implementation-checks-{step:06d}.json',result);print('IMPLEMENTATION_CHECKS_PASS',step,len(checks),'networks; max intervention error',max(r['column_vs_input_intervention_max_abs'] for r in checks))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-step',type=int,default=100000);args=ap.parse_args();main(args.checkpoint_step)
