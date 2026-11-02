"""Measure pair geometry of the exact, previously checked posterior denoisers."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import characterization_core as B
import coupled_characterization as K
import analyze_toy_cis as F


def main():
    output = B.ROOT / 'reports/toy-characterization-oracle-geometry-v1'; output.mkdir(exist_ok=True)
    assert not (output / 'analysis.json').exists()
    rows = []; grids = []; predictions = []
    axis = np.linspace(0, 1.5, 31); aa, bb = np.meshgrid(axis, axis, indexing='ij')
    for family, C in [('orthogonal', B), ('coupled', K)]:
        with np.load(C.OUT / 'analysis-inputs.npz') as data: assets = dict(data)
        for pair in [[0,1], [4,9]]:
            for sign in [[1,1], [1,-1]]:
                coeff = np.stack([sign[0]*aa.ravel(), sign[1]*bb.ravel()], -1)
                actual = []; null = []; interaction = []
                for bg in range(4):
                    context = float(assets['background_context'][bg]); matrix = C.effective_encoder(assets['encoder'], 'linear_denoise', context)
                    z = assets['background_bits'][bg].copy(); z[pair] = 0; base = (matrix @ z)[None]
                    def f(x): return C.posterior_mean(x, assets['encoder'], np.full(len(x), context), 'linear_denoise', .35)[1]
                    y0 = f(base); y1 = f(base + coeff[:, :1] @ matrix[:, pair[:1]].T)
                    y2 = f(base + coeff[:, 1:] @ matrix[:, pair[1:]].T)
                    y = f(base + coeff @ matrix[:, pair].T); predicted = y1 + y2 - y0
                    response = np.linalg.norm(y-y0, axis=-1).reshape(31,31)
                    actual.append(response); null.append(np.linalg.norm(predicted-y0, axis=-1).reshape(31,31))
                    interaction.append(float(np.linalg.norm(y-predicted, axis=-1).mean()/response.max()))
                actual = np.stack(actual); null = np.stack(null)
                fits = {str(fr): F.fit(np.median(actual,axis=0),fr)[0] for fr in [.25,.5,.75]}
                null_fits = {str(fr): F.fit(np.median(null,axis=0),fr)[0] for fr in [.25,.5,.75]}
                rows.append({'family':family,'pair':pair,'sign':sign,'fits':fits,'vector_additive_fits':null_fits,
                             'mean_vector_interaction_residual':float(np.mean(interaction))})
                grids.append(actual); predictions.append(null)
    np.savez_compressed(output/'responses.npz',response=np.stack(grids),vector_null=np.stack(predictions))
    B.write(output/'analysis.json',{'records':rows,'source_sha256':B.sha(__file__),
             'protocol_sha256':B.sha(B.ROOT/'configs/toy-characterization-oracle-geometry-v1.json'),
             'raw_sha256':B.sha(output/'responses.npz')})
    fig,axes=plt.subplots(1,2,figsize=(9,4.8))
    for ax,family in zip(axes,['orthogonal','coupled']):
        for pair,color,label in [([0,1],'#9364a4','Pair (0,1)'),([4,9],'#377bad','Pair (4,9)')]:
            index=next(i for i,r in enumerate(rows) if r['family']==family and r['pair']==pair and r['sign']==[1,1])
            fit=rows[index]['fits']['0.5'];pts=F.F.extract_contour(axis,np.median(grids[index],axis=0),fit['threshold'])
            pts/=np.array(fit['intercepts_feature_units']);pts=pts[np.argsort(np.arctan2(pts[:,1],pts[:,0]))]
            suffix = 'fit flagged' if fit['flags'] else f'p={fit["p"]:.2f}'
            ax.plot(pts[:,0],pts[:,1],color=color,lw=2,label=f'{label}: {suffix}')
        ax.set(title='Independent code' if family=='orthogonal' else 'Overlapping code',xlabel='First coefficient / axis cutoff',ylabel='Second coefficient / axis cutoff',aspect='equal',xlim=(0,1.1),ylim=(0,1.1))
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('An exact denoiser has different geometry for different probe pairs',fontsize=13)
    fig.tight_layout(rect=(0,.15,1,.92));fig.text(.5,.025,'Measured contours, same exact denoiser and noise level. A flagged fit is not summarized as a reliable exponent.\nPair (0,1) has input cosine 0.6 in the overlap code: this diagnostic lies outside the paper’s filter.',ha='center',fontsize=9)
    fig.savefig(output/'boundaries.png',dpi=170,bbox_inches='tight');plt.close(fig)
    for row in rows:print(row['family'],row['pair'],row['sign'],'p',row['fits']['0.5']['p'],'flags',row['fits']['0.5']['flags'])


if __name__=='__main__':main()
