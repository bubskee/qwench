"""Generate descriptive seed summaries; do not treat pair fits as independent trials."""
from pathlib import Path
import json,statistics
import scalar_comparison_runtime as R

def describe(values):
 values=[v for v in values if v is not None]
 return {'n':len(values),'mean':statistics.mean(values) if values else None,'min':min(values) if values else None,'max':max(values) if values else None,'sd':statistics.stdev(values) if len(values)>1 else None}

def main():
 models=json.loads((R.OUT/'model-summary.json').read_text());results=[]
 for arch in R.P['architectures']:
  for loss in R.P['loss_exponents']:
   rows=[r for r in models if r['architecture']==arch['name'] and r['loss_exponent']==loss];assert len(rows)==3
   result={'architecture':arch['name'],'activation':arch['activation'],'loss_exponent':loss,'seeds':[r['seed'] for r in rows]}
   for metric in ['competent_features','mean_per_feature_positive_mse','inactive_rmse','mean_competent_feature_participation','mean_neuron_feature_participation','feature_error_cv']:
    result[metric]=describe([r[metric] for r in rows])
   for cohort in ['fixed','fixed_competent','competence_conditioned']:
    for kind in ['encoder','random']:
     for anchor in ['zero','positive','negative']:
      field=f'{cohort}_{kind}_{anchor}_median_p';result[field]=describe([r[field] for r in rows]);result[f'{cohort}_{kind}_{anchor}_flagged_pair_fits']=sum(r[f'{cohort}_{kind}_{anchor}_flagged'] for r in rows)
   results.append(result)
 R.write(R.OUT/'seed-summary.json',{'groups':results,'scope':'Means/ranges describe three initialization seeds on a fixed dataset. p entries summarize network-level pair medians; no population significance claim.'})
 def fmt(v,precision=3):return '—' if v is None else f'{v:.{precision}f}'
 lines=['# Generated numerical summary','', '| Activation | Loss | Competent features, seed range | Mean active MSE | Mean feature reuse per unit | p, absent pair | p, negative pair |','|---|---:|---:|---:|---:|---:|---:|']
 for r in results:
  coverage=r['competent_features'];p0=r['competence_conditioned_encoder_zero_median_p'];pn=r['competence_conditioned_encoder_negative_median_p']
  lines.append('| '+ ' | '.join([r['architecture'],f'L{r["loss_exponent"]}',f'{coverage["min"]}–{coverage["max"]}',fmt(r['mean_per_feature_positive_mse']['mean'],5),fmt(r['mean_neuron_feature_participation']['mean'],2),fmt(p0['mean'])+f' ({p0["n"]}/3 nets)',fmt(pn['mean'])+f' ({pn["n"]}/3 nets)'])+' |')
 lines+=['','Geometry entries use encoder columns, ++ sign and competence-conditioned pairs. These are point summaries; flag counts, all signs and task-ineligible fixed pairs remain in the full records.','']
 (R.OUT/'numerical-summary.md').write_text('\n'.join(lines));print('\n'.join(lines))

if __name__=='__main__':main()
