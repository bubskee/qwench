"""Compare original Qwen3-1.7B cached measurements with the fresh 4B pilot."""
from pathlib import Path
import argparse, hashlib, json, pickle, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_stress_4b as A

ROOT=Path(__file__).resolve().parents[1]


def stats(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return {'n':len(x),'median':float(np.median(x)),'mean':float(np.mean(x)),
            'count_gt_2':int((x>2).sum()),'fraction_gt_2':float((x>2).mean()),
            'min':float(x.min()),'max':float(x.max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT.parents[1]/'work/fsec-reference')
    args=parser.parse_args();ref=args.reference.resolve()
    out=ROOT/'reports/qwen17-comparison';out.mkdir(exist_ok=True)
    scratch=ROOT.parents[1]/'work/qwen17-paper';scratch.mkdir(exist_ok=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ref,text=True).strip()
    assert revision=='a9355d4001500708a40a255bacfe559e6dc269c0'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','results/sweeps_2d'],cwd=ref,text=True).splitlines()
    paths=sorted(p for p in paths if Path(p).name.startswith('sweep2d_qwen_L2_random_') and p.endswith('_fineweb_60deg_dirrandom.pkl'))
    random_dirs=pickle.loads((ref/'results/directions/dirs_qwen_L2_random.pkl').read_bytes())['directions']
    controls=[]
    for p in paths:
        raw=subprocess.check_output(['git','show','HEAD:'+p],cwd=ref)
        local=scratch/Path(p).name;local.write_bytes(raw)
        x=pickle.loads(raw);angles=x['angles_deg'];grid=np.median(x['l2'],axis=0)
        assert x['n_anchors']==30 and angles.tolist()==list(range(0,61,2))
        a,b=x['direction_labels'];overlap=abs(float(random_dirs[a]@random_dirs[b]))
        assert overlap<.1
        fit=A.fit(angles,grid,A.threshold(grid))
        controls.append({'path':p,'sha256':hashlib.sha256(raw).hexdigest(),
                         'pair':list(x['direction_labels']),'fit':fit,
                         'raw_absolute_cosine':overlap,
                         'origin_l2_max':float(x['l2'][:,0,0].max()),
                         'n_anchors':x['n_anchors'],'angles_deg':angles.tolist(),
                         'model_full':x['model_full'],'perturb_layer':x['perturb_layer'],'measure_layer':x['measure_layer']})
    blob=pickle.loads((ref/'results/directions/dirs_qwen_L2.pkl').read_bytes())
    dirs={k:v.cpu().numpy().astype(np.float64) for k,v in blob['directions'].items()}
    fits=pickle.loads((ref/'results/fits/fits_qwen_L2_thrpair_exact.pkl').read_bytes())
    excluded={'Formal','HonestyShort','TensePresent'}
    feature_map={}
    for names,v in fits.items():
        if set(names)&excluded or abs(dirs[names[0]]@dirs[names[1]])>.1:
            continue
        p=v[('1.0xT',60.)]['p_median']
        if np.isfinite(p):feature_map[frozenset(names)]={'pair':list(names),'p':float(p)}
    features=list(feature_map.values())
    gemma=pickle.loads((ref/'results/directions/dirs_gemma_L2.pkl').read_bytes())['directions']
    figure_features=[r for r in features if abs(float(gemma[r['pair'][0]]@gemma[r['pair'][1]]))<=.1]
    assert len(features)==326 and len(figure_features)==310 and len(controls)==len(paths)
    four_dir=ROOT/'reports/stress-4b-v1/analysis'
    four=[json.loads(p.read_text()) for p in sorted(four_dir.glob('*_primary.json'))]
    lookup={frozenset(r['pair']):r['p'] for r in features}
    matched=[{'pair':[r['a'],r['b']],'qwen17_cached_p':lookup[frozenset([r['a'],r['b']])],
              'qwen35_4b_fresh_p':r['headline']['p']}
             for r in four if r['kind']=='paper' and frozenset([r['a'],r['b']]) in lookup]
    cohorts={
      'qwen17_original_features':stats([r['p'] for r in figure_features]),
      'qwen17_own_model_overlap_only':stats([r['p'] for r in features]),
      'qwen17_original_isotropic_controls':stats([r['fit']['p'] for r in controls]),
      'qwen35_4b_paper_features':stats([r['headline']['p'] for r in four if r['kind']=='paper']),
      'qwen35_4b_isotropic_controls':stats([r['headline']['p'] for r in four if r['stratum']=='isotropic_random']),
      'qwen17_common_named_pairs':stats([r['qwen17_cached_p'] for r in matched]),
      'qwen35_4b_common_named_pairs':stats([r['qwen35_4b_fresh_p'] for r in matched]),
    }
    result={'scope':'Original 1.7B data reanalysis versus fresh 4B adapted pilot; no fresh 1.7B model forwards.',
            'reference_revision':revision,'cohorts':cohorts,'matched_pairs':matched,
            'original_feature_fits':figure_features,'qwen_own_overlap_feature_fits':features,'original_random_control_refits':controls,
            'caveats':['Different anchor corpus/count, grid, pair cohort, hardware/runtime, and model architecture. No causal attribution to model identity.',
                       'Only isotropic controls are compared here. No Qwen-specific random-activation-difference fit cache is present in the pinned release.',
                       'Original raw control data contain the recorded origin responses; no numerical-floor correction is applied.',
                       'The common-pair comparison uses the deterministic intersection of eligible named contrasts, not matched latent directions.']}
    result['caveats'].append('The published-figure plotting path filters model columns by their own overlaps and then by the target Gemma overlaps. This gives 310 Qwen pairs; the Qwen-only cutoff gives 326. Ordered historical duplicates are collapsed with the original last-entry-wins rule.')
    A.write(out/'comparison.json',result)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for i,(label,vals,color) in enumerate([
        ('Original 1.7B\nfeatures',[r['p'] for r in figure_features],'#397ca0'),
        ('Original 1.7B\nrandom',[r['fit']['p'] for r in controls],'#82939b'),
        ('Fresh 4B\npaper features',[r['headline']['p'] for r in four if r['kind']=='paper'],'#317b67'),
        ('Fresh 4B\nrandom',[r['headline']['p'] for r in four if r['stratum']=='isotropic_random'],'#777777')]):
        jitter=np.random.default_rng(117).uniform(-.2,.2,len(vals))
        axes[0].scatter(i+jitter,vals,s=12,alpha=.55,color=color)
        axes[0].plot([i-.27,i+.27],[np.median(vals)]*2,color=color,linewidth=2)
    axes[0].set(xticks=range(4),xticklabels=['Original 1.7B\nfeatures','Original 1.7B\nrandom','Fresh 4B\npaper features','Fresh 4B\nrandom'],ylabel='Fitted p',title='Original cached 1.7B data versus adapted 4B pilot')
    axes[0].tick_params(axis='x',labelsize=9)
    axes[1].scatter([r['qwen17_cached_p'] for r in matched],[r['qwen35_4b_fresh_p'] for r in matched],color='#317b67')
    axes[1].plot([1,4],[1,4],':',color='#999999')
    axes[1].set(xlabel='Original Qwen3-1.7B cached p',ylabel='Fresh Qwen3.5-4B p',title=f'Same eligible named contrasts ({len(matched)} pairs)')
    for ax in axes:
        ax.axhline(2,color='#bd713a',linestyle='--');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('Feature distribution transfer and control comparison — settings differ',fontsize=13)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170,bbox_inches='tight');fig.savefig(out/'comparison.pdf',bbox_inches='tight')
    print(json.dumps(A.clean({'cohorts':cohorts,'origin_l2_max':max(r['origin_l2_max'] for r in controls)}),indent=2))

if __name__=='__main__':main()
