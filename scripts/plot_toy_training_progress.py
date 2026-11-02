"""Render a clearly provisional view from saved held-out training checks."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_trained_toys import OUT,P,COLORS,LABELS

def main():
 snaps=[json.loads(p.read_text()) for p in sorted(OUT.glob('eval-*.json'))];latest=snaps[-1];names=[a['name'] for a in P['architectures']]
 fig,axes=plt.subplots(1,2,figsize=(11,4.3))
 for name in names:
  selected=[r for r in latest['networks'] if r['architecture']==name and r['loss_exponent']==4]
  curves=np.array([np.sort(r['per_feature_positive_mse']) for r in selected])
  axes[0].plot(np.arange(1,101),curves.mean(0),color=COLORS[name],label=LABELS[name])
  coverage=[]
  for snap in snaps:coverage.append(np.mean([r['competent_features'] for r in snap['networks'] if r['architecture']==name and r['loss_exponent']==4]))
  axes[1].plot([s['step'] for s in snaps],coverage,color=COLORS[name],label=LABELS[name],marker='.',ms=5)
 axes[0].axhline(.05,color='#aaa',ls='--');axes[0].set(xlabel='Feature rank, sorted within each network',ylabel='Positive-active MSE',title='L4 loss: current held-out feature errors');axes[0].legend(fontsize=8)
 axes[1].set(xlabel='Training updates',ylabel='Accurately computed features / 100',title='L4 loss: declared competence cutoff',ylim=(-2,102),xlim=(0,100000))
 for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.suptitle(f'Provisional training check: {latest["step"]:,} / 100,000 updates\nThree initialization seeds per configuration. This is not the completed comparison.',fontsize=12)
 fig.tight_layout();path=OUT/f'progress-{latest["step"]:06d}.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig);print(path)

if __name__=='__main__':main()
