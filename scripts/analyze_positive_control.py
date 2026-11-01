"""Analyze the fixed fresh 1.7B / 4B comparison without model inference."""
from __future__ import annotations
import argparse,hashlib,json,pickle,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_stress_4b as A

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/positive-control-v1'
REF=ROOT.parents[1]/'work/fsec-reference'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_grid(angles,grid,bootstrap):
    median=np.median(grid,axis=0);base=A.threshold(median)
    main=A.fit(angles,median,base)
    if median[0,0]>=base:main.setdefault('flags',[]).append('origin_at_or_above_threshold')
    rng=np.random.default_rng(423);draws=[]
    for _ in range(bootstrap):
        sample=np.median(grid[rng.integers(0,len(grid),size=len(grid))],axis=0)
        draws.append(A.fit(angles,sample,A.threshold(sample))['p'])
    valid=np.asarray(draws);valid=valid[np.isfinite(valid)]
    return A.clean({'fit':main,'bootstrap_p':draws,
                    'anchor_interval_90':np.percentile(valid,[5,95]).tolist() if len(valid) else None,
                    'bootstrap_valid':len(valid),'bootstrap_requested':bootstrap,
                    'threshold_sensitivity':{str(f):A.fit(angles,median,base*f) for f in [.5,1,2]},
                    'per_anchor':[A.fit(angles,g,A.threshold(g)) for g in grid],
                    'origin_l2_median':float(np.median(grid[:,0,0])),
                    'origin_l2_max':float(grid[:,0,0].max()),
                    'origin_median_over_threshold':float(np.median(grid[:,0,0])/base)})


def contour(angles,grid,result):
    fit=result['fit']
    if fit['p'] is None:return None
    t1,t2=fit['intercepts'];points=A.F.extract_contour(angles,np.median(grid,axis=0),fit['threshold'])
    points=points[(points[:,0]<angles.max()-1)&(points[:,1]<angles.max()-1)]
    radius=np.hypot(points[:,0],points[:,1]);safe=np.where(radius>1e-12,radius,1.)
    return np.sin(np.deg2rad(radius))[:,None]*points/safe[:,None]/np.sin(np.deg2rad([t1,t2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args();out=RUN/'analysis';out.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
    source_hash=sha(Path(__file__));records=[];arrays={}
    for pair in protocol['pairs']:
        reference_path=out/('reference_'+pair['id']+'.npz')
        if not reference_path.exists():
            filename=f"results/sweeps_2d/sweep2d_qwen_L2_{pair['a']}__{pair['b']}_fineweb_60deg"+('_dirrandom' if pair['kind']=='isotropic' else '')+'.pkl'
            raw=subprocess.check_output(['git','show',protocol['reference_revision']+':'+filename],cwd=REF)
            original=pickle.loads(raw)
            np.savez_compressed(reference_path,angles_deg=original['angles_deg'],l2=original['l2'])
            A.write(reference_path.with_suffix('.json'),{'path':filename,'raw_sha256':hashlib.sha256(raw).hexdigest(),
                    'reference_revision':protocol['reference_revision'],'anchor_norm_mean':original['anchor_norm_mean'],
                    'measure_layer':original['measure_layer'],'model_full':original['model_full']})
        candidates=[('original_qwen17','paper',reference_path,'l2')]
        for model in protocol['models']:
            path=RUN/model['key']/(pair['id']+'.npz')
            if not path.exists():continue
            data=np.load(path)
            if not data['completed_anchors'].all():continue
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['complete'] and receipt['matched_origin_max']==0 and receipt['norm_error_max']<.005
            for mode in ['paper','matched']:candidates.append((model['key'],mode,path,'l2_'+mode))
        for model,mode,path,key in candidates:
            name=model+'_'+pair['id']+'_'+mode;target=out/(name+'.json');raw_hash=sha(path)
            record=json.loads(target.read_text()) if target.exists() else None
            data=np.load(path);arrays[name]=(data['angles_deg'],data[key])
            if not record or record['analysis_source_sha256']!=source_hash or record['bootstrap_requested']!=args.bootstrap or record['raw_sha256']!=raw_hash:
                record={'model':model,'pair':pair,'mode':mode,'raw_sha256':raw_hash,'analysis_source_sha256':source_hash,
                        **fit_grid(data['angles_deg'],data[key],args.bootstrap)}
                A.write(target,record);print('FIT',name,record['fit']['p'],flush=True)
            records.append(record)
    lookup={(r['model'],r['pair']['id'],r['mode']):r for r in records}
    differences=[]
    for pair in protocol['pairs']:
        for mode in ['paper','matched']:
            a=lookup.get(('qwen17',pair['id'],mode));b=lookup.get(('qwen35_4b',pair['id'],mode))
            if not a or not b:continue
            draws=np.array([np.nan if x is None else x for x in b['bootstrap_p']])-np.array([np.nan if x is None else x for x in a['bootstrap_p']])
            valid=draws[np.isfinite(draws)]
            differences.append({'pair':pair['id'],'mode':mode,'delta_p_4b_minus_17':b['fit']['p']-a['fit']['p'] if b['fit']['p'] is not None and a['fit']['p'] is not None else None,
                                'paired_anchor_interval_90':np.percentile(valid,[5,95]).tolist() if len(valid) else None})
    complete=all((RUN/m['key']/'COMPLETE.json').exists() for m in protocol['models'])
    summary={'complete':complete,'records':records,'cross_model_differences':differences,
             'analysis_source_sha256':source_hash,'bootstrap':args.bootstrap,
             'scope':'Two fixed feature pairs and one fixed isotropic pair per model. Procedural positive control and matched transfer; no population-level feature-specificity inference.',
             'anchor_identity':'Regenerated pinned FineWeb-Edu seed-42 recipe; historical original token identity unverified.'}
    A.write(out/'summary.json',summary)
    rows=[]
    for pair in protocol['pairs']:
        for model,mode in [('original_qwen17','paper'),('qwen17','paper'),('qwen17','matched'),('qwen35_4b','paper'),('qwen35_4b','matched')]:
            r=lookup.get((model,pair['id'],mode))
            if r:rows.append(f"{pair['id']},{model},{mode},{r['fit']['p']},{r['fit']['mean_radial_frac']},{r['origin_l2_max']}")
    (out/'results.csv').write_text('pair,model,reference_mode,p,mean_radial_fraction,origin_l2_max\n'+'\n'.join(rows)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    styles=[('original_qwen17','paper','Cached\n1.7B','#888888'),('qwen17','paper','Fresh\n1.7B','#3875a4'),('qwen17','matched','1.7B\nmatched','#87a9bf'),('qwen35_4b','paper','Fresh\n4B','#317b67'),('qwen35_4b','matched','4B\nmatched','#8abbad')]
    for ax,pair in zip(axes,protocol['pairs']):
        for i,(model,mode,label,color) in enumerate(styles):
            r=lookup.get((model,pair['id'],mode))
            if not r or r['fit']['p'] is None:continue
            value=r['fit']['p'];interval=r['anchor_interval_90']
            if interval:ax.plot([i,i],interval,color=color,linewidth=1.5)
            ax.scatter(i,value,color=color,s=45,marker='D' if mode=='paper' else 'o',zorder=3)
        ax.axhline(2,color='#bd713c',linestyle='--')
        ax.set(xticks=range(5),xticklabels=[s[2] for s in styles],ylabel='Fitted p',title=pair['a']+' × '+pair['b'])
        ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Fresh positive control and matched transfer: original reference versus batch-matched check\n30 FineWeb-Edu anchors, 2° component grid; intervals resample anchors (90%).',fontsize=12)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5))
    for ax,pair in zip(axes,protocol['pairs']):
        for model,mode,label,color in [styles[i] for i in [0,1,3]]:
            r=lookup.get((model,pair['id'],mode))
            if not r:continue
            angles,grid=arrays[model+'_'+pair['id']+'_'+mode];points=contour(angles,grid,r)
            if points is not None:ax.plot(points[:,0],points[:,1],color=color,label=label.replace('\n',' ')+f" p={r['fit']['p']:.2f}",linewidth=1.8)
        x,y=A.F.superellipse_curve(2);ax.plot(x,y,':',color='#aaaaaa')
        ax.set(xlim=(0,1.25),ylim=(0,1.25),aspect='equal',title=pair['a']+' × '+pair['b'],xlabel='Normalized first component',ylabel='Normalized second component')
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.suptitle('Measured normalized boundaries — single-anchor reference used by original code',fontsize=12)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'contours.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    print('ANALYSIS',len(records),'records; complete',complete,flush=True)

if __name__=='__main__':main()
