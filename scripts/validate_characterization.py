"""Final fresh-noise validation and matched neighboring-feature interventions."""
import argparse
import json
import numpy as np
import torch
import characterization_core as B
import analyze_characterization as A
from diagnose_characterization import project

ROOT = B.ROOT
OUT = ROOT / 'reports/toy-characterization-validation-v1'
PROTOCOL = ROOT / 'configs/toy-characterization-validation-v1.json'
P = json.loads(PROTOCOL.read_text())


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'inputs.npz').exists()
    r = P['recovery']; rng = np.random.default_rng(r['data_seed'])
    bits = (rng.random((r['samples'],16)) < .125).astype(float)
    context = rng.choice([-1.,1.],r['samples'])
    noise = np.random.default_rng(r['noise_seed']).normal(size=(2,r['samples'],32))
    np.savez_compressed(OUT/'inputs.npz',bits=bits,context=context,noise=noise)
    B.write(OUT/'prepared.json',{'protocol_sha256':B.sha(PROTOCOL),'inputs_sha256':B.sha(OUT/'inputs.npz'),
                               'source_sha256':B.sha(__file__),'samples':r['samples'],'noise_draws':2})


def measure(family):
    if family == 'orthogonal': C = B; null = A.separable
    else:
        import coupled_characterization as C
        null = C.separable
    A.C = C; A.separable = null
    assert not (OUT/(family+'.json')).exists()
    with np.load(OUT/'inputs.npz') as values: bits=values['bits'];context=values['context'];noise=values['noise']
    with np.load(C.OUT/'analysis-inputs.npz') as values: assets=dict(values)
    metas=[row for row in json.loads((C.OUT/'analysis-plan.json').read_text())['networks'] if row['track']=='new' and row['step']==100000]
    sums={}; all_count=np.full(len(bits),16); active_count=(bits>0).sum(-1);inactive_count=(bits==0).sum(-1)
    recovery=[];geometry=[];grids=[];max_control_state_error=0.
    for meta in metas:
        model=A.model_from(meta);enc=torch.tensor(assets['encoder'])
        base=C.task_inputs(torch.tensor(bits),torch.tensor(context),enc,meta['task'])
        corrupt=base[None].expand(2,*base.shape).clone();corrupt[...,:32]+=P['recovery']['noise_std']*torch.tensor(noise)
        x=corrupt.reshape(-1,33);xp=project(x,assets['encoder'],meta['task'])
        f=C.forward(x,model,meta['activation']).numpy().reshape(2,*bits.shape)
        fp=C.forward(xp,model,meta['activation']).numpy().reshape(2,*bits.shape)
        fs=np.concatenate([null(batch,model,meta,assets['encoder']).numpy() for batch in x.split(256)]).reshape(2,*bits.shape)
        clean_f=C.forward(base,model,meta['activation']).numpy()
        clean_s=np.concatenate([null(batch,model,meta,assets['encoder']).numpy() for batch in base.split(256)])
        maps={'natural':f,'projected':fp,'separable':fs,'clipped_natural':np.clip(f,0,1),
              'clipped_projected':np.clip(fp,0,1),'clipped_separable':np.clip(fs,0,1)}
        clean_maps={'natural':clean_f,'projected':clean_f,'separable':clean_s,
                    'clipped_natural':np.clip(clean_f,0,1),'clipped_projected':np.clip(clean_f,0,1),'clipped_separable':np.clip(clean_s,0,1)}
        linear=[];oracle=[]
        for draw in range(2):
            lr,post=C.posterior_mean(corrupt[draw,:,:32].numpy(),assets['encoder'],context,meta['task'],.35)
            linear.append(lr);oracle.append(post)
        maps['linear_reference']=np.stack(linear);maps['Bayes_oracle']=np.stack(oracle)
        for name,prediction in maps.items():
            error=(prediction-bits)**2;row={'network':meta,'family':family,'map':name}
            for label,mask,count in [('all',np.ones_like(bits,dtype=bool),all_count),('active',bits>0,active_count),('inactive',bits==0,inactive_count)]:
                value=(error*mask).sum(-1).mean(0);sums[meta['key']+'_'+name+'_'+label+'_sum']=value
                row[label+'_mse']=float(value.sum()/count.sum())
                if name in clean_maps:row['clean_'+label+'_mse']=float((((clean_maps[name]-bits)**2)*mask).sum()/count.sum())
            recovery.append(row)
        control_states=[]
        for state in [0,1]:
            adjusted={key:value.copy() for key,value in assets.items()}
            adjusted['background_bits'][:,[5,8]]=state
            for which in ['natural','separable']:
                response=A.geometry_case(model,meta,adjusted,[4,9],0,[1,1],'feature_axes',which,31)
                if which=='separable':control_states.append(response)
                fits={str(fr):A.FIT.fit(np.median(response,axis=0),fr)[0] for fr in [.25,.5,.75]}
                geometry.append({'network':meta,'family':family,'pair':[4,9],'partner_state':state,'map':which,'fits':fits})
                grids.append(response)
        state_error=float(np.abs(control_states[0]-control_states[1]).max());max_control_state_error=max(max_control_state_error,state_error)
        assert state_error<1e-8
        print('VALIDATED',family,meta['key'],flush=True)
    np.savez_compressed(OUT/(family+'-errors.npz'),all_count=all_count,active_count=active_count,inactive_count=inactive_count,**sums)
    np.savez_compressed(OUT/(family+'-geometry.npz'),response=np.stack(grids))
    B.write(OUT/(family+'.json'),{'recovery':recovery,'geometry':geometry,
            'raw_errors_sha256':B.sha(OUT/(family+'-errors.npz')),'raw_geometry_sha256':B.sha(OUT/(family+'-geometry.npz')),
            'control_partner_state_invariance_max_error':max_control_state_error,'all_control_checks_passed':True,
            'source_sha256':B.sha(__file__),'protocol_sha256':B.sha(PROTOCOL),'inputs_sha256':B.sha(OUT/'inputs.npz')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','measure'],required=True)
    parser.add_argument('--family',choices=['orthogonal','coupled'],default='orthogonal')
    args=parser.parse_args()
    with torch.inference_mode():prepare() if args.stage=='prepare' else measure(args.family)
