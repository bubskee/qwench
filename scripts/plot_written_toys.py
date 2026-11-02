"""Plot the saved, fresh-evaluation tradeoffs; no inference or fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(out):
    rows = json.loads((out/'metrics.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    fig.suptitle('Can formula-written weights compete?', fontsize=17, fontweight='bold')
    colors = {'written': '#007d79', 'trained': '#b65d22'}
    ax = axes[0]
    groups = [(name, [r for r in rows if r['name'] == name]) for name in ['axis50', 'code5_gain09', 'code10_gain09']]
    for loss in [2, 4]:
        groups.append((f'Trained ReLU L{loss}', [r for r in rows if r['track']=='bottleneck' and r['origin']=='trained' and r['activation']=='relu' and r['loss_exponent']==loss]))
    offsets = {'axis50': (7, 8), 'code5_gain09': (-100, -13), 'code10_gain09': (-100, 9),
               'Trained ReLU L2': (7, -13), 'Trained ReLU L4': (-115, 16)}
    for name, group in groups:
        x = np.mean([r['inactive_rmse'] for r in group]); y = np.mean([r['positive_mse'] for r in group])
        origin = group[0]['origin']
        ax.scatter(x, y, s=75, color=colors[origin], marker='o' if origin=='written' else 's', zorder=3)
        ax.annotate(name, (x, y), xytext=offsets[name], textcoords='offset points', fontsize=9)
    ax.set(xlabel='Inactive-output RMSE (lower is better)', ylabel='Positive-feature MSE (lower is better)',
           title='Sparse task: accuracy / leakage tradeoff', xlim=(-.005,.065), ylim=(-.012,.19))
    ax.grid(alpha=.2)
    ax = axes[1]
    labels=[]; written=[]; trained=[]; oracle=[]
    for family in ['orthogonal', 'coupled']:
        for task in ['linear_denoise', 'context_denoise']:
            group=[r for r in rows if r.get('family')==family and r.get('task')==task and r.get('sigma')==.35]
            written.append(np.mean([r['all_mse'] for r in group if r['origin']=='written']))
            trained.append(np.mean([r['all_mse'] for r in group if r['origin']=='trained' and r['activation']=='relu']))
            oracle.append(np.mean([r['all_mse'] for r in group if r['origin']=='oracle']))
            labels.append(('Independent' if family=='orthogonal' else 'Overlapping')+'\n'+('simple' if task.startswith('linear') else 'context'))
    x=np.arange(4)
    ax.bar(x-.18,written,width=.36,color=colors['written'],label='Written weights')
    ax.bar(x+.18,trained,width=.36,color=colors['trained'],label='Trained ReLU (3 seeds)')
    ax.scatter(x,oracle,marker='_',s=220,color='#20232a',linewidths=2,label='Exact posterior',zorder=4)
    ax.set(xticks=x,xticklabels=labels,ylabel='All-output MSE (lower is better)',ylim=(0,.065),title='Binary denoising at σ = 0.35')
    ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,-.03,'Same architectures and raw-output evaluation • 49,152 fresh examples per condition • Constructors know the task',ha='center',fontsize=9,color='#555555')
    fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    plot(parser.parse_args().out)
