"""Summarize fixed implementation and legacy-anchor A/A comparisons."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_stress_4b as A
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/implementation-audit-v1';PRIOR=ROOT/'reports/positive-control-v1'
protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())

def record(path,key):
 data=np.load(path);grid=data[key];angles=data['angles_deg'];median=np.median(grid,axis=0)
 return {'path':str(path.relative_to(ROOT)),'key':key,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'fit':A.fit(angles,median,A.threshold(median))},grid,angles
summary={'pairs':[],'implementation':[]}
for name in ['fresh_directions','released_directions']:
 path=OUT/(name+'.npz')
 if path.exists():
  row,_,_=record(path,'l2');summary['implementation'].append({'case':name,**row})
fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),sharey=True)
for ax,pair in zip(axes,protocol['pairs']):
 ident=pair['id'];candidates=[('Published cache',PRIOR/'analysis'/('reference_'+ident+'.npz'),'l2','#666666'),
  ('Our first run\ndatasets 5.1',PRIOR/'qwen17'/(ident+'.npz'),'l2_paper','#b86b4f'),
  ('Older sampler\ndatasets 4.8.5',OUT/('legacy_'+ident+'.npz'),'l2','#337c69')]
 rows=[];grids=[]
 for i,(label,path,key,color) in enumerate(candidates):
  if not path.exists():continue
  row,grid,angles=record(path,key);row['label']=label;rows.append(row);grids.append(grid)
  p=row['fit']['p'];ax.scatter(i,p,s=65,color=color,zorder=3);ax.annotate(f'{p:.3f}',(i,p),xytext=(0,8),textcoords='offset points',ha='center',color=color)
 ax.axhline(2,ls='--',color='#aaa');ax.set(xticks=range(3),xticklabels=[c[0] for c in candidates],title=pair['a']+' × '+pair['b'],ylabel='Fitted p',ylim=(1.5,2.7))
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelsize=8)
 result={'pair':pair,'records':rows}
 if len(grids)==3:
  original,new=grids[0],grids[2]
  diff=np.abs(new-original)
  # Full 31x31 response fingerprints, not just the fitted exponent.
  x=original.reshape(30,-1);y=new.reshape(30,-1)
  rms=np.sqrt(((x[:,None,:]-y[None,:,:])**2).mean(-1))
  corr=np.corrcoef(x,y)[:30,30:]
  result['legacy_vs_original']={'mean_absolute_l2_difference':float(diff.mean()),'max_absolute_l2_difference':float(diff.max()),
   'mean_abs_difference_over_original_mean':float(diff.mean()/original.mean()),
   'per_anchor_response_correlation':np.diag(corr).tolist(),
   'same_index_nearest_response_surface_count':int((rms.argmin(1)==np.arange(30)).sum()),
   'nearest_legacy_anchor_indices':rms.argmin(1).tolist(),
   'p_difference':rows[2]['fit']['p']-rows[0]['fit']['p']}
 summary['pairs'].append(result)
fig.suptitle('Qwen3-1.7B audit: changing the data sampler restores the cached geometry\nSame three fixed pairs; 30 anchors, 2° grid, original reference measurement',fontsize=12)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/('sampler-comparison.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
A.write(OUT/'fit-summary.json',summary)
for row in summary['implementation']:print('IMPLEMENTATION',row['case'],row['fit']['p'])
for row in summary['pairs']:
 print('PAIR',row['pair']['id'],[(r['label'].replace('\n',' '),r['fit']['p']) for r in row['records']])
 if 'legacy_vs_original' in row:print('IDENTITY',row['legacy_vs_original'])
