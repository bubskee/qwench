"""Small fresh Qwen 1.7B → 4B replication with frozen FineWeb inputs."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.metadata,json,pickle,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/positive-control-v1'
REF=ROOT.parents[1]/'work/fsec-reference'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


class Captured(Exception):pass


class Engine:
    def __init__(self,pin,protocol):
        self.pin,self.protocol=pin,protocol
        self.layer=protocol['direction_layer'];self.measure=pin['measure_layer']
        torch.set_num_threads(6);torch.manual_seed(42)
        assert torch.backends.mps.is_available()
        config=AutoConfig.from_pretrained(pin['id'],revision=pin['revision'],local_files_only=True).get_text_config()
        self.model=AutoModelForCausalLM.from_pretrained(pin['id'],revision=pin['revision'],config=config,
                    dtype=torch.bfloat16,attn_implementation=protocol['attention_implementation'],local_files_only=True).to('mps').eval()
        self.model.requires_grad_(False)
        self.tokenizer=AutoTokenizer.from_pretrained(pin['id'],revision=pin['revision'],local_files_only=True)
        self.d_model=config.hidden_size;self.fixed={};self.baselines={}
        assert self.measure==config.num_hidden_layers-2

    @torch.inference_mode()
    def capture(self,ids,layer,full=False,all_tokens=False):
        saved={}
        def hook(module,args,out):
            assert torch.is_tensor(out)
            saved['value']=(out if all_tokens else out[:,-1]).detach().clone()
            if not full:raise Captured
        handle=self.model.model.layers[layer].register_forward_hook(hook)
        try:
            try:self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            except Captured:pass
        finally:handle.remove()
        return saved['value']

    def tokens(self,ids,batch=1):
        return torch.tensor([ids],device='mps').expand(batch,-1).contiguous()

    def install_anchor(self,index,ids,hidden):
        self.fixed[index]=(ids,hidden.to(device='mps',dtype=torch.bfloat16))

    def changed_hidden(self,index,batch,d1=None,d2=None,coords=None):
        ids,h=self.fixed[index]
        out=h.expand(batch,-1,-1).clone()
        if coords is None:return out,0.
        a=h[0,-1].float();unit=a/a.norm()
        u=d1-(unit@d1)*unit;u=u/u.norm()
        v=d2-(unit@d2)*unit;v=v-(u@v)*u
        assert float(v.norm())>1e-6
        v=v/v.norm()
        alpha=torch.deg2rad(torch.tensor(coords,device='mps',dtype=torch.float32))
        radius=alpha.square().sum(-1).sqrt();safe=torch.where(radius>1e-12,radius,torch.ones_like(radius))
        tangent=(alpha[:,:1]*u+alpha[:,1:]*v)/safe[:,None]
        point=torch.cos(radius)[:,None]*a+torch.sin(radius)[:,None]*a.norm()*tangent
        point=torch.where((radius==0)[:,None],a,point)
        out[:,-1]=point.to(out.dtype)
        norm_error=float((out[:,-1].float().norm(dim=-1)/a.norm()-1).abs().max())
        assert norm_error<.005
        return out,norm_error

    @torch.inference_mode()
    def replay(self,index,batch,changed=None):
        ids,h=self.fixed[index];saved={}
        value=h.expand(batch,-1,-1).clone() if changed is None else changed
        def inject(module,args,out):return value
        def measure(module,args,out):
            saved['value']=out[:,-1].float().clone();raise Captured
        handles=[self.model.model.layers[self.layer].register_forward_hook(inject),
                 self.model.model.layers[self.measure].register_forward_hook(measure)]
        try:
            try:self.model.model(input_ids=self.tokens(ids,batch),attention_mask=torch.ones((batch,len(ids)),device='mps',dtype=torch.long),use_cache=False)
            except Captured:pass
        finally:
            for handle in handles:handle.remove()
        return saved['value']

    @torch.inference_mode()
    def batch(self,index,d1,d2,coords):
        count=len(coords)
        for n in {1,count}:
            if (index,n) not in self.baselines:self.baselines[index,n]=self.replay(index,n)
        changed,error=self.changed_hidden(index,count,d1,d2,coords)
        output=self.replay(index,count,changed)
        paper=(output-self.baselines[index,1]).norm(dim=-1).cpu().numpy()
        matched=(output-self.baselines[index,count]).norm(dim=-1).cpu().numpy()
        assert np.isfinite(paper).all() and np.isfinite(matched).all()
        zero=np.linalg.norm(np.asarray(coords),axis=1)==0
        assert not zero.any() or float(matched[zero].max())==0.,'Matched origin parity failed'
        return paper,matched,error

    @torch.inference_mode()
    def validate(self,index,d1,d2):
        ids,h=self.fixed[index]
        checks={}
        for layer in [self.layer,self.measure]:
            a=self.capture(self.tokens(ids),layer);b=self.capture(self.tokens(ids),layer,full=True)
            checks[f'stop_full_layer_{layer}']=float((a.float()-b.float()).abs().max())
            assert checks[f'stop_full_layer_{layer}']==0
        native=self.capture(self.tokens(ids),self.measure).float();replay=self.replay(index,1)
        checks['fixed_prefix_replay_vs_native']=float((native-replay).abs().max())
        assert checks['fixed_prefix_replay_vs_native']==0
        if self.pin['key']=='qwen17':
            parity=[]
            for batch in [1,8]:
                for coordinate in [(0.,0.),(8.,6.)]:
                    changed,_=self.changed_hidden(index,batch,d1,d2,[coordinate]*batch)
                    expected=self.replay(index,batch,changed)
                    hidden=changed
                    pos=torch.arange(hidden.shape[1],device='mps')[None]
                    embeddings=self.model.model.rotary_emb(hidden,pos)
                    for block in self.model.model.layers[self.layer+1:self.measure+1]:
                        hidden=block(hidden,position_embeddings=embeddings)
                        if isinstance(hidden,tuple):hidden=hidden[0]
                    error=float((hidden[:,-1].float()-expected).abs().max())
                    parity.append({'batch':batch,'coordinates':coordinate,'max_absolute_error':error})
                    assert error==0.,('Original suffix/native replay mismatch',parity[-1])
            checks['original_suffix_parity']=parity
        coords=[(float(i%11),float((3*i)%11)) for i in range(64)]
        self.batch(index,d1,d2,coords)
        torch.mps.synchronize();start=time.perf_counter()
        for _ in range(3):self.batch(index,d1,d2,coords)
        torch.mps.synchronize();seconds=time.perf_counter()-start
        checks['timing']={'examples':192,'seconds':seconds,'examples_per_second':192/seconds,
                          'estimated_seconds_per_30_anchor_surface':30*31*31/(192/seconds)}
        return checks


def prepare(engine,protocol,seal,out):
    bank=json.loads((ROOT/'data/stress-features-v1.json').read_text())['directions']
    bank={r['name']:r for r in bank}
    names=sorted({p[k] for p in protocol['pairs'] if p['kind']=='contrastive' for k in ['a','b']})
    directions={};meta={}
    released=pickle.loads((REF/'results/directions/dirs_qwen_L2.pkl').read_bytes())['prompt_set_hashes']
    for name in names:
        for side in ['a','b']:
            digest=hashlib.sha256()
            for prompt in bank[name][side]:digest.update(prompt.encode());digest.update(b'\x00')
            assert digest.hexdigest()==released[name][side+'_sha']
        states={}
        for side in ['a','b']:
            values=[]
            for prompt in bank[name][side]:
                ids=engine.tokenizer(prompt)['input_ids']
                values.append(engine.capture(engine.tokens(ids),engine.layer)[0].float().cpu())
            states[side]=torch.stack(values)
        vector=(states['b']-states['a']).mean(0);directions[name]=F.normalize(vector,dim=0)
        meta[name]={'pairs':len(states['a']),'raw_norm':float(vector.norm()),
                    'released_prompt_hashes_match':True,
                    'prompts_a_sha256':hashlib.sha256(json.dumps(bank[name]['a'],ensure_ascii=False).encode()).hexdigest(),
                    'prompts_b_sha256':hashlib.sha256(json.dumps(bank[name]['b'],ensure_ascii=False).encode()).hexdigest()}
        print('DIRECTION',engine.pin['key'],name,flush=True)
    generator=torch.Generator().manual_seed(protocol['random_seed'])
    random=F.normalize(torch.randn(protocol['random_direction_count'],engine.d_model,generator=generator),dim=-1)
    for i in range(2):directions[f'random_{i:03d}']=random[i]
    original_cosines={}
    if engine.pin['key']=='qwen17':
        for filename,selected in [('dirs_qwen_L2.pkl',names),('dirs_qwen_L2_random.pkl',['random_000','random_001'])]:
            original=pickle.loads((REF/'results/directions'/filename).read_bytes())['directions']
            for name in selected:original_cosines[name]=float(directions[name]@original[name])
    anchors=json.loads((RUN/'anchors.json').read_text())['anchors']
    hidden={}
    for i,anchor in enumerate(anchors):
        ids=anchor['models'][engine.pin['key']]['token_ids']
        h=engine.capture(engine.tokens(ids),engine.layer,all_tokens=True).cpu()
        hidden[f'anchor_{i:02d}']=h
        engine.install_anchor(i,ids,h)
    pairs=[]
    for pair in protocol['pairs']:
        cosine=float(directions[pair['a']]@directions[pair['b']])
        pairs.append({**pair,'raw_cosine':cosine,'eligible':abs(cosine)<protocol['max_absolute_cosine']})
    d1=directions[protocol['pairs'][0]['a']].to('mps');d2=directions[protocol['pairs'][0]['b']].to('mps')
    validation=engine.validate(0,d1,d2)
    save_file(directions,str(out/'directions.safetensors'))
    save_file(hidden,str(out/'fixed-prefixes.safetensors'))
    write(out/'prepared.json',{'seal':seal,'model':engine.pin,'pairs':pairs,'directions':meta,
          'original_direction_cosines':original_cosines,'validation':validation,
          'mean_anchor_norm':float(torch.stack([h[0,-1].float().norm() for h in hidden.values()]).mean()),
          'directions_sha256':sha(out/'directions.safetensors'),'prefixes_sha256':sha(out/'fixed-prefixes.safetensors')})
    print('PREPARED',engine.pin['key'],json.dumps(validation['timing']),flush=True)


def sweep(engine,protocol,seal,out,source_hash):
    prepared=json.loads((out/'prepared.json').read_text());assert prepared['seal']==seal
    assert sha(out/'directions.safetensors')==prepared['directions_sha256']
    assert sha(out/'fixed-prefixes.safetensors')==prepared['prefixes_sha256']
    directions=load_file(str(out/'directions.safetensors'));prefixes=load_file(str(out/'fixed-prefixes.safetensors'))
    anchors=json.loads((RUN/'anchors.json').read_text())['anchors']
    for i,anchor in enumerate(anchors):engine.install_anchor(i,anchor['models'][engine.pin['key']]['token_ids'],prefixes[f'anchor_{i:02d}'])
    angles=protocol['angles_deg'];coordinates=[(a,b) for a in angles for b in angles];n=len(angles)
    started=time.perf_counter();prior_seconds=0.
    for pair in prepared['pairs']:
        if not pair['eligible']:
            print('EXCLUDED',engine.pin['key'],pair['id'],pair['raw_cosine'],flush=True);continue
        path=out/(pair['id']+'.npz');receipt=out/(pair['id']+'.json')
        paper=np.full((30,n,n),np.nan,dtype=np.float32);matched=paper.copy();completed=np.zeros(30,dtype=bool);norms=np.zeros(30,dtype=np.float32)
        seconds=np.zeros(30,dtype=np.float64)
        if path.exists():
            saved=np.load(path);prior=json.loads(receipt.read_text());assert prior['seal']==seal and prior['pair']==pair
            assert prior['source_sha256']==source_hash
            paper,matched,completed,norms,seconds=[saved[k].copy() for k in ['l2_paper','l2_matched','completed_anchors','norm_error','seconds']]
        d1=directions[pair['a']].to('mps');d2=directions[pair['b']].to('mps')
        for i in range(30):
            if completed[i]:continue
            start=time.perf_counter();one=[];two=[];max_error=0.
            for st in range(0,len(coordinates),protocol['batch_size']):
                p,m,error=engine.batch(i,d1,d2,coordinates[st:st+protocol['batch_size']])
                one.extend(p.tolist());two.extend(m.tolist());max_error=max(max_error,error)
            paper[i]=np.array(one).reshape(n,n);matched[i]=np.array(two).reshape(n,n)
            norms[i]=max_error;seconds[i]=time.perf_counter()-start;completed[i]=True
            with path.with_suffix('.tmp').open('wb') as file:
                np.savez_compressed(file,angles_deg=np.asarray(angles),l2_paper=paper,l2_matched=matched,
                                    completed_anchors=completed,norm_error=norms,seconds=seconds)
            path.with_suffix('.tmp').replace(path)
            write(receipt,{'seal':seal,'source_sha256':source_hash,'model':engine.pin,'pair':pair,
                  'completed_anchors':int(completed.sum()),'complete':bool(completed.all()),
                  'matched_origin_max':float(matched[completed,0,0].max()),
                  'paper_origin_max':float(paper[completed,0,0].max()),'norm_error_max':float(norms.max()),
                  'seconds':float(seconds.sum())})
            progress={'model':engine.pin['key'],'pair':pair['id'],'completed_anchors':int(completed.sum()),
                      'anchor_seconds':float(seconds[i]),'elapsed_this_process':time.perf_counter()-started}
            write(RUN/'progress.json',progress);print('ANCHOR',json.dumps(progress),flush=True)
    write(out/'COMPLETE.json',{'seal':seal,'source_sha256':source_hash,'eligible_pairs':sum(p['eligible'] for p in prepared['pairs'])})
    print('MODEL_COMPLETE',engine.pin['key'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','sweep'],required=True)
    parser.add_argument('--model',choices=['qwen17','qwen35_4b','both'],default='both')
    args=parser.parse_args();protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
    seal={'protocol_sha256':sha(ROOT/'configs/positive-control-v1.json'),'anchors_sha256':sha(RUN/'anchors.json'),
          'feature_bank_sha256':sha(ROOT/'data/stress-features-v1.json'),'reference_revision':protocol['reference_revision']}
    if (RUN/'input-seal.json').exists():assert json.loads((RUN/'input-seal.json').read_text())==seal
    else:write(RUN/'input-seal.json',seal)
    source=Path(__file__).read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    (RUN/f'{args.stage}-source-{source_hash[:12]}.py').write_bytes(source)
    packages={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','safetensors']}
    for pin in protocol['models']:
        if args.model not in ['both',pin['key']]:continue
        out=RUN/pin['key'];out.mkdir(exist_ok=True)
        if args.stage=='prepare' and (out/'prepared.json').exists():
            assert json.loads((out/'prepared.json').read_text())['seal']==seal
            print('PREPARATION_ALREADY_COMPLETE',pin['key'],flush=True);continue
        engine=Engine(pin,protocol)
        if args.stage=='prepare':prepare(engine,protocol,seal,out)
        else:sweep(engine,protocol,seal,out,source_hash)
        write(out/(args.stage+'-receipt.json'),{'seal':seal,'source_sha256':source_hash,'packages':packages})
        del engine;gc.collect();torch.mps.empty_cache()

if __name__=='__main__':main()
