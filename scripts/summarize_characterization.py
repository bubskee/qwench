"""Seed-level descriptive tables for all fixed toy-characterization conditions."""
from pathlib import Path
import csv
import json
import statistics as S
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/toy-characterization-study-v1'
FAMILIES = {'orthogonal': 'toy-characterization-v1', 'coupled': 'toy-characterization-coupled-v1'}


def median(values):
    values = [v for v in values if v is not None]
    return S.median(values) if values else None


def describe(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'mean': S.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def main():
    OUT.mkdir(exist_ok=True)
    records = []; fine = []; quality = {}; monotonicity = []
    for family, folder in FAMILIES.items():
        source = ROOT / 'reports' / folder
        plan = json.loads((source / 'analysis-plan.json').read_text())
        files = sorted((source / 'analysis').glob('*/*.json'))
        assert len(files) == len(plan['networks'])
        for path in files:
            data = json.loads(path.read_text()); meta = data['network']
            if meta['track'] == 'new' and meta['step'] == 100000 and meta['task'].endswith('denoise'):
                for kind in ['feature_axes', 'isotropic_observation_random']:
                    selected_quality = [(i, r['fits']['0.5']) for i, r in enumerate(data['geometry'])
                                        if r['kind'] == kind and r['anchor'] == 0 and r['sign'] == [1,1] and r['map'] == 'natural']
                    quality.setdefault(family, {}).setdefault(kind, []).extend(f for i, f in selected_quality)
                    if kind == 'feature_axes':
                        with np.load(path.with_suffix('.npz')) as raw:
                            for index, fit in selected_quality:
                                if 'nonmonotone_axis' not in fit['flags']: continue
                                grid = np.median(raw['response'][index], axis=0)
                                for axis_curve in [grid[:,0],grid[0,:]]:
                                    crossing = int(np.flatnonzero(axis_curve >= fit['threshold'])[0])
                                    differences = np.diff(axis_curve)
                                    monotonicity.append({'family':family,'key':meta['key'],
                                        'reversal_before_first_crossing':bool((differences[:crossing] < -1e-8).any()),
                                        'returns_below_threshold':bool((axis_curve[crossing:] < fit['threshold']-1e-8).any()),
                                        'max_single_step_drop_fraction':float(max(0.,-differences.min())/axis_curve.max())})
            label = family if meta['track'] == 'new' else 'historical'
            row = {key: meta[key] for key in ['key', 'activation', 'task', 'seed', 'step']}
            row['family'] = label; row['competent_features'] = data['task_evaluation']['competent_features']
            for which in ['natural', 'separable']:
                selected = [r['fits']['0.5'] for r in data['geometry'] if r['kind'] == 'feature_axes' and r['anchor'] == 0 and r['sign'] == [1, 1] and r['map'] == which]
                good = [r['p'] for r in selected if not r['flags']]
                row[which + '_p'] = median([r['p'] for r in selected])
                row[which + '_p_unflagged'] = median(good)
                row[which + '_good_fits'] = len(good)
                row[which + '_flagged_fits'] = sum(bool(r['flags']) for r in selected)
                row[which + '_radial_residual'] = median([r.get('mean_radial_frac') for r in selected])
                random_fits = [r['fits']['0.5'] for r in data['geometry'] if r['kind'] == 'isotropic_observation_random'
                               and r['anchor'] == 0 and r['sign'] == [1, 1] and r['map'] == which]
                row[which + '_random_p'] = median([r['p'] for r in random_fits])
                row[which + '_random_good_fits'] = sum(not r['flags'] for r in random_fits)
                row[which + '_random_flagged_fits'] = sum(bool(r['flags']) for r in random_fits)
                scores = [r for r in data['recovery'] if r['map'] == which]
                sigma = .075 if label == 'historical' else .35
                clean = next(r for r in scores if r['sigma'] == 0)
                noisy = next(r for r in scores if r['sigma'] == sigma)
                for metric in ['all_mse', 'active_mse', 'inactive_mse']:
                    row[which + '_' + metric] = noisy[metric]
                    row[which + '_clean_' + metric] = clean[metric]
                    row[which + '_added_' + metric] = noisy[metric] - clean[metric]
                row[which + '_miss_rate'] = noisy['active_miss_rate']
                row[which + '_false_positive_rate'] = noisy['inactive_false_positive_rate']
            row['p_minus_separable'] = row['natural_p'] - row['separable_p'] if row['natural_p'] is not None and row['separable_p'] is not None else None
            row['recovery_advantage_over_separable'] = row['separable_all_mse'] - row['natural_all_mse']
            row['clean_adjusted_advantage'] = row['separable_added_all_mse'] - row['natural_added_all_mse']
            reference_name = 'task_reference' if label == 'historical' else 'linear_reference'
            reference = next(r for r in data['recovery'] if r['map'] == reference_name and r['sigma'] == sigma)
            row['reference_mse'] = reference['all_mse']
            row['Bayes_mse'] = next((r['all_mse'] for r in data['recovery'] if r['map'] == 'Bayes_oracle' and r['sigma'] == sigma), None)
            denominator = row['reference_mse'] - row['Bayes_mse'] if row['Bayes_mse'] is not None else None
            row['oracle_available_improvement_fraction'] = (row['reference_mse'] - row['natural_all_mse']) / denominator if denominator is not None and denominator > 1e-12 else None
            records.append(row)
            fine.extend([{'family': label, **meta, **r} for r in data['fine_checks']])
    keys = list(records[0])
    with (OUT / 'networks.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    (OUT / 'networks.json').write_text(json.dumps(records, indent=2) + '\n')
    metrics = [key for key in keys if key not in ['key', 'family', 'activation', 'task', 'seed', 'step']]
    groups = []
    for identity in sorted(set((r['family'], r['activation'], r['task'], r['step']) for r in records)):
        family, activation, task, step = identity
        selected = [r for r in records if (r['family'], r['activation'], r['task'], r['step']) == identity]
        assert len(selected) == 3
        groups.append({'family': family, 'activation': activation, 'task': task, 'step': step,
                       'statistics': {metric: describe([r[metric] for r in selected]) for metric in metrics}})
    (OUT / 'seed-summary.json').write_text(json.dumps(groups, indent=2) + '\n')
    (OUT / 'fine-checks.json').write_text(json.dumps(fine, indent=2) + '\n')
    quality_summary = {}
    for family, kinds in quality.items():
        quality_summary[family] = {}
        for kind, fits in kinds.items():
            quality_summary[family][kind] = {'attempted':len(fits),'unflagged':sum(not f['flags'] for f in fits),
                'flags':dict(Counter(flag for f in fits for flag in f['flags'])),
                'median_superellipse_residual':median([f.get('mean_radial_frac') for f in fits]),
                'median_quadratic_residual':median([f['quadratic']['mean_radial_frac'] for f in fits if 'quadratic' in f])}
    quality_summary['flag_location_diagnostic'] = {'axes_checked':len(monotonicity),
        'reversals_before_threshold_crossing':sum(r['reversal_before_first_crossing'] for r in monotonicity),
        'returns_below_threshold':sum(r['returns_below_threshold'] for r in monotonicity),
        'max_single_step_drop_fraction':max(r['max_single_step_drop_fraction'] for r in monotonicity)}
    (OUT / 'fit-quality.json').write_text(json.dumps(quality_summary,indent=2)+'\n')
    (OUT / 'monotonicity-diagnostics.json').write_text(json.dumps(monotonicity,indent=2)+'\n')
    lines = ['# Final checkpoint descriptive summaries', '',
             'Mean over three initialization seeds; p is each network’s median across two fixed pairs.',
             'Nominal p includes flagged fits; the good-fit column shows how many of the six pair fits are unflagged.', '',
             '| Family | Task | Activation | p | Null p | Good fits / 6 | Noisy MSE | Null MSE | Bayes MSE |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value): return '—' if value is None else f'{value:.4f}'
    for group in groups:
        if group['step'] != 100000: continue
        stats = group['statistics']
        lines.append('| ' + ' | '.join([group['family'], group['task'], group['activation'],
                     fmt(stats['natural_p']['mean']), fmt(stats['separable_p']['mean']),
                     str(int(stats['natural_good_fits']['mean'] * 3)), fmt(stats['natural_all_mse']['mean']),
                     fmt(stats['separable_all_mse']['mean']), fmt(stats['Bayes_mse']['mean'])]) + ' |')
    lines += ['', 'Noise std: 0.35 for new coded-bit tasks, 0.075 for the separate historical signed-sparse task.',
              'The analytic null knows the true feature coordinates and preserves the trained isolated response curves.',
              'See projection-matched diagnostics and per-seed CSV before assigning a mechanistic interpretation.', '']
    (OUT / 'numerical-summary.md').write_text('\n'.join(lines))
    print('SUMMARIZED', len(records), 'network/checkpoint frames;', len(groups), 'seed groups; fine checks', len(fine))


if __name__ == '__main__': main()
