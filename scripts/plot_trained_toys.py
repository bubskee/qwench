"""Plot task competence, functional sharing, and geometry separately."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import importlib,os
T=importlib.import_module(os.environ.get('QWENCH_TRAINED_DRIVER','train_scalar_core'))
ROOT=T.ROOT;OUT=T.OUT;AN=OUT/'analysis';P=T.P
COLORS={'relu50':'#666666','gelu50':'#377bad','silu50':'#368866','swiglu33':'#b66d3e','swiglu50':'#9364a4','leakyrelu50':'#b66d3e','relu2_50':'#9364a4'}
LABELS={'relu50':'ReLU 50','gelu50':'GELU 50','silu50':'SiLU 50','swiglu33':'SwiGLU 33','swiglu50':'SwiGLU 50','leakyrelu50':'Leaky ReLU 0.1','relu2_50':'Squared ReLU'}

def main():
 snapshots=[json.loads(p.read_text()) for p in sorted(OUT.glob('eval-*.json'))]
 assert snapshots and snapshots[-1]['step']==100000
 final=snapshots[-1]['networks'];cohorts=json.loads((AN/'cohorts.json').read_text());names=[a['name'] for a in P['architectures']]
 fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey='col')
 for ri,exp in enumerate([2,4]):
  for col,(metric,title) in enumerate([('competent_features','Accurately computed features'),('mean_per_feature_positive_mse','Positive-active MSE, all features'),('mean_neuron_feature_participation','Functional sharing among competent features')]):
   ax=axes[ri,col]
   for name in names:
    curves=[]
    for seed in P['seeds']:
     values=[next(r[metric] for r in snap['networks'] if r['architecture']==name and r['loss_exponent']==exp and r['seed']==seed) for snap in snapshots]
     vals=np.array([np.nan if v is None else v for v in values]);curves.append(vals);ax.plot([s['step'] for s in snapshots],vals,color=COLORS[name],alpha=.25,lw=.7)
    arr=np.asarray(curves);valid=np.isfinite(arr).sum(0);mean=np.nansum(arr,axis=0)/np.where(valid,valid,1);mean[valid==0]=np.nan
    ax.plot([s['step'] for s in snapshots],mean,color=COLORS[name],label=LABELS[name],lw=1.8)
   ax.set(title=f'L{exp}: '+title,xlabel='Training updates',ylabel='Features / 100' if col==0 else 'MSE' if col==1 else 'Effective features per hidden output');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
   if col==0:ax.set_ylim(0,102)
   if col==2:ax.set_ylim(0,12)
 axes[0,0].legend(fontsize=8);fig.suptitle('Training from scratch: three initialization seeds per architecture/objective\nIdentical data/minibatches, 100,000 updates each. Bold: mean over seeds; faint: individual seeds.',fontsize=12)
 fig.tight_layout();fig.savefig(OUT/'training-and-sharing.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
 for ri,exp in enumerate([2,4]):
  for ci,anchor in enumerate([0,.5,-.5]):
   ax=axes[ri,ci]
   for i,name in enumerate(names):
    for kind,offset,marker in [('encoder_columns',-.09,'o'),('gram_matched_random',.09,'x')]:
     selected=[r for r in cohorts if r['architecture']==name and r['loss_exponent']==exp and r['cohort']=='competence_conditioned' and r['kind']==kind and r['quadrant']=='++' and r['anchor_value']==anchor]
     for r in selected:
      if r['median_p'] is None:continue
      x=i+offset+(r['seed']-1)*.035
      ax.scatter(x,r['median_p'],s=40,color=COLORS[name],marker=marker,alpha=.8) if marker=='x' else ax.scatter(x,r['median_p'],s=40,facecolors='none' if r['flagged'] else COLORS[name],edgecolors=COLORS[name],marker=marker,alpha=.8)
   ax.axhline(2,color='#aaa',ls='--');ax.set(title=f'L{exp}, pair inputs {anchor:+.1f}',xticks=range(len(names)),xticklabels=[LABELS[n].replace(' ','\n') for n in names],ylabel='Median fitted p, ++ sign');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 axes[0,0].legend(handles=[Line2D([0],[0],color='#444',ls='',marker='o',label='Encoder columns'),Line2D([0],[0],color='#444',ls='',marker='x',label='Gram-matched random'),Line2D([0],[0],color='#444',ls='',marker='o',markerfacecolor='none',label='Any pair fit flagged')],fontsize=8)
 fig.suptitle('Geometry on three task-competent pairs per trained network\nEach point: one network; selection uses task accuracy before p. Missing points mean no valid eligible fit. All flags retained in data.',fontsize=12)
 fig.tight_layout();fig.savefig(OUT/'geometry-comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.6))
 for ax,anchor in zip(axes,[0,-.5]):
  for r in final:
   record=next(c for c in cohorts if c['id']==r['id'] and c['cohort']=='competence_conditioned' and c['kind']=='encoder_columns' and c['quadrant']=='++' and c['anchor_value']==anchor)
   if record['median_p'] is None or r['mean_neuron_feature_participation'] is None:continue
   ax.scatter(r['mean_neuron_feature_participation'],record['median_p'],facecolors='none' if record['flagged'] else COLORS[r['architecture']],edgecolors=COLORS[r['architecture']],marker='o' if r['loss_exponent']==2 else 'D',s=40,alpha=.8)
  ax.axhline(2,ls='--',color='#aaa');ax.set(title=f'Pair inputs {anchor:+.1f}',xlabel='Effective competent features per neuron',ylabel='Median fitted p');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.suptitle('Functional sharing and geometry are separate measurements\nOne point per network with a competent-feature fit; circles L2, diamonds L4; hollow markers include flagged pair fits.',fontsize=12)
 fig.legend(handles=[Line2D([0],[0],color=COLORS[n],ls='',marker='o',label=LABELS[n]) for n in names],loc='lower center',ncol=len(names),fontsize=8)
 fig.tight_layout(rect=(0,.09,1,1));fig.savefig(OUT/'sharing-versus-p.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for ax,exp in zip(axes,[2,4]):
  for name in names:
   values=np.array([np.sort(r['per_feature_positive_mse']) for r in final if r['architecture']==name and r['loss_exponent']==exp])
   for v in values:ax.plot(np.arange(1,101),v,color=COLORS[name],lw=.7,alpha=.25)
   ax.plot(np.arange(1,101),values.mean(0),color=COLORS[name],label=LABELS[name],lw=1.8)
  ax.axhline(.05,color='#aaa',ls='--',label='Competence MSE cutoff');ax.set(title=f'L{exp}: all feature errors',xlabel='Feature rank, sorted within each network',ylabel='Positive-active MSE');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'feature-errors.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,len(names),figsize=(max(10,3.2*len(names)),6.8),sharey=True)
 for ri,exp in enumerate([2,4]):
  for ci,name in enumerate(names):
   ax=axes[ri,ci]
   for r in final:
    if r['architecture']==name and r['loss_exponent']==exp:ax.scatter(r['feature_contribution_participation'],r['per_feature_positive_mse'],s=9,alpha=.35,color=COLORS[name])
   ax.axhline(.05,color='#aaa',ls='--');ax.set(title=f'{LABELS[name]}, L{exp}',xlabel='Effective contributing units per feature',ylabel='Positive-active MSE' if ci==0 else '');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
 fig.suptitle('Task fidelity and distributed contributions, feature by feature\nAll 100 features from all three seeds; dashed line is the declared competence threshold. Distributed contributions alone are not CiS proof.',fontsize=12);fig.tight_layout();fig.savefig(OUT/'feature-sharing-and-error.png',dpi=170,bbox_inches='tight');plt.close(fig)
 records=[]
 for row in final:
  result={k:row[k] for k in ['id','architecture','activation','width','parameters','loss_exponent','seed','competent_features','mean_per_feature_positive_mse','inactive_rmse','inactive_false_positive_rate','isolated_active_mse','multi_input_active_mse','mean_competent_feature_participation','mean_neuron_feature_participation']}
  per=np.asarray(row['per_feature_positive_mse']);result['feature_error_cv']=float(per.std()/max(per.mean(),1e-20))
  for cohort in ['fixed','fixed_competent','competence_conditioned']:
   for kind,short in [('encoder_columns','encoder'),('gram_matched_random','random')]:
    for anchor,tag in [(0,'zero'),(.5,'positive'),(-.5,'negative')]:
     c=next(r for r in cohorts if r['id']==row['id'] and r['cohort']==cohort and r['kind']==kind and r['quadrant']=='++' and r['anchor_value']==anchor)
     for key in ['median_p','finite','flagged','median_radial_residual']:result[f'{cohort}_{short}_{tag}_{key}']=c[key]
  records.append(result)
 with (OUT/'models.csv').open('w') as file:
  writer=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator="\n");writer.writeheader();writer.writerows(records)
 (OUT/'model-summary.json').write_text(json.dumps(records,indent=2)+'\n')
 print('PLOTS_COMPLETE',len(records),'networks',flush=True)

if __name__=='__main__':main()
