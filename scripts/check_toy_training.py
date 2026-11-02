"""Verify independent-network gradients, paired initialization and architecture counts."""
from pathlib import Path
import json,torch
import train_toy_activations as T

def main():
 torch.set_num_threads(2);groups,params=T.initialize('cpu');g=torch.Generator().manual_seed(8989)
 x=(torch.rand(256,100,generator=g)*2-1)*(torch.rand(256,100,generator=g)<.02);target=x.relu();rows=[]
 for a in T.P['architectures']:
  group=groups[a['name']];pred=T.predict(x,group,a['activation']);exponents=torch.tensor([e for e,s in T.network_ids()])[:,None,None]
  losses=(pred-target).abs().pow(exponents).mean((1,2));losses.sum().backward()
  errors=[];ratios=[]
  for i,(exponent,seed) in enumerate(T.network_ids()):
   one={role:w[i].detach().clone().requires_grad_() for role,w in group.items()}
   loss=(T.predict(x,one,a['activation'])-target).abs().pow(exponent).mean();loss.backward()
   for role,w in one.items():
    error=float((w.grad-group[role].grad[i]).abs().max());errors.append(error)
    assert torch.allclose(w.grad,group[role].grad[i],atol=1e-10,rtol=1e-5),(a['name'],i,role,error)
    if float(w.grad.norm())>0:ratios.append(float(group[role].grad[i].norm()/w.grad.norm()))
  count=sum(w[0].numel() for w in group.values());assert count==a['parameters']
  for w in group.values():assert torch.equal(w[:3],w[3:])
  independent=True
  if a['activation']=='swiglu':independent=not torch.equal(group['gate'],group['value']);assert independent
  rows.append({'architecture':a['name'],'per_network_parameters':count,'max_independent_vs_batched_gradient_error':max(errors),'gradient_norm_ratio_range':[min(ratios),max(ratios)],'paired_objective_initialization_exact':True,'gate_value_independent':independent})
 result={'all_assertions_passed':True,'architectures':rows,'claim':'Joint training sums per-network losses; no seed or objective gradient dilution. Proper SwiGLU has independent trainable branches.'}
 T.write(T.OUT/'preflight-checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
