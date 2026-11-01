"""Analyze every completed fresh 4B stress surface, including failures and controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fsec_superellipse', ROOT / 'vendor/fsec/scripts/lib/superellipse.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def fit(angles, grid, threshold):
    if threshold <= 0 or min(grid[:, 0].max(), grid[0, :].max()) < threshold:
        return {'p': np.nan, 'mean_radial_frac': np.nan, 'n_pts': 0, 'failure': 'axis_threshold_not_reached'}
    result = F.fit_p_one_pass(angles, grid, threshold, float(angles.max()), exact_geodesic=True)
    result['threshold'] = float(threshold)
    result['flags'] = []
    if not np.isfinite(result['p']):
        result['flags'].append('no_usable_fit')
    else:
        if result['p'] < .501 or result['p'] > 9.99:
            result['flags'].append('parameter_bound')
        if result['mean_radial_frac'] > .02:
            result['flags'].append('radial_residual_over_2pct')
    return result


def threshold(grid, fraction=.5):
    return fraction * min(float(grid[:, 0].max()), float(grid[0, :].max()))


def finite_ps(rows):
    return np.array([r['headline']['p'] for r in rows if r['headline']['p'] is not None and np.isfinite(r['headline']['p'])])


def summarize(rows):
    ps = finite_ps(rows)
    return {'attempted': len(rows), 'finite': len(ps), 'invalid': len(rows) - len(ps),
            'mean': float(ps.mean()) if len(ps) else np.nan,
            'median': float(np.median(ps)) if len(ps) else np.nan,
            'fraction_gt_2': float((ps > 2).mean()) if len(ps) else np.nan,
            'flagged': sum(bool(r['headline'].get('flags')) for r in rows)}


def direction_bootstrap(rows, iterations=2000, seed=764):
    rows = [r for r in rows if r['headline']['p'] is not None]
    if not rows:
        return {}
    names = sorted({r[k] for r in rows for k in ['a', 'b']})
    indices = {name: i for i, name in enumerate(names)}
    a = np.array([indices[r['a']] for r in rows]); b = np.array([indices[r['b']] for r in rows])
    values = np.array([r['headline']['p'] for r in rows])
    order = np.argsort(values); means, medians = [], []
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        multiplicity = rng.multinomial(len(names), np.full(len(names), 1 / len(names)))
        weights = multiplicity[a] * multiplicity[b]
        if weights.sum() == 0:
            continue
        means.append(float(np.average(values, weights=weights)))
        cumulative = np.cumsum(weights[order])
        medians.append(float(values[order[np.searchsorted(cumulative, weights.sum() / 2)]]))
    return {'valid_draws': len(means), 'iterations': iterations, 'direction_count': len(names),
            'mean_interval_95': np.percentile(means, [2.5, 97.5]).tolist(),
            'median_interval_95': np.percentile(medians, [2.5, 97.5]).tolist(),
            'scope': 'Direction-cluster resampling conditional on the fixed tested edge cohort; not a population-of-models interval.'}


def analyze_surface(path, metadata, bootstrap_count):
    data = np.load(path, allow_pickle=False)
    angles, grid = data['angles_deg'], data['l2']
    median = np.median(grid, axis=0)
    base = threshold(median)
    headline = fit(angles, median, base)
    result = {**metadata['task'], 'file': str(path), 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'headline': headline, 'quality': {'zero_angle_l2_max': metadata['zero_angle_l2_max'],
                                             'max_relative_norm_error': metadata['max_relative_norm_error']},
              'threshold_sensitivity': {str(factor): fit(angles, median, factor * base) for factor in [.5, 1., 2.]}}
    per_anchor, loo = [], []
    for i, individual in enumerate(grid):
        per_anchor.append({'anchor_index': metadata['task']['anchor_bank'][i],
                           'own_threshold_fit': fit(angles, individual, threshold(individual)),
                           'shared_threshold_fit': fit(angles, individual, base)})
        if len(grid) > 1:
            reduced = np.median(np.delete(grid, i, axis=0), axis=0)
            loo.append(fit(angles, reduced, threshold(reduced)))
    result['per_anchor'] = per_anchor
    result['leave_one_anchor_out'] = loo
    if metadata['task']['tag'] == 'primary' and bootstrap_count:
        rng = np.random.default_rng(423)
        values = []
        for _ in range(bootstrap_count):
            indices = rng.integers(0, len(grid), size=len(grid))
            sample = np.median(grid[indices], axis=0)
            f = fit(angles, sample, threshold(sample))
            if np.isfinite(f['p']):
                values.append(f['p'])
        result['anchor_bootstrap'] = {'requested': bootstrap_count, 'valid': len(values),
                                      'interval_90': np.percentile(values, [5, 95]).tolist() if values else None,
                                      'p_draws': values,
                                      'threshold_recalibrated_in_each_draw': True}
    # Compare an equally parameterized tilted quadratic on the same usable points.
    if np.isfinite(headline['p']):
        raw = F.extract_contour(angles, median, base)
        raw = raw[(raw[:, 0] < angles.max() - 1) & (raw[:, 1] < angles.max() - 1)]
        t1, t2 = headline['intercepts']
        radius = np.hypot(raw[:, 0], raw[:, 1]); safe = np.where(radius > 1e-12, radius, 1.)
        x = np.sin(np.deg2rad(radius)) * raw[:, 0] / safe / np.sin(np.deg2rad(t1))
        y = np.sin(np.deg2rad(radius)) * raw[:, 1] / safe / np.sin(np.deg2rad(t2))
        keep = (x > .05) & (y > .05) & (x < 1.5) & (y < 1.5)
        x, y = x[keep], y[keep]
        cross = 2 * x * y
        delta = -float(np.sum((x*x + y*y - 1) * cross) / np.sum(cross*cross))
        residual = float(np.abs(np.sqrt(np.clip(x*x + 2*delta*x*y + y*y, 1e-9, None)) - 1).mean())
        result['tilted_ellipse'] = {'delta': delta, 'positive_definite': abs(delta) < 1,
                                     'radial_residual': residual,
                                     'lower_residual_than_superellipse': residual < headline['mean_radial_frac']}
    return clean(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/stress-4b-v1')
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--reference-dir', type=Path, default=ROOT.parents[1] / 'work/fsec-reference')
    args = parser.parse_args()
    run = args.run.resolve(); out = run / 'analysis'; out.mkdir(exist_ok=True)
    plan = json.loads((run / 'plan.json').read_text())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    for task in plan['tasks']:
        path = run / 'surfaces' / (task['id'] + '.npz')
        metadata_path = path.with_suffix('.json')
        if not (path.exists() and metadata_path.exists()):
            continue
        target = out / (task['id'] + '.json')
        metadata = json.loads(metadata_path.read_text())
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(target.read_text()) if target.exists() else None
        if not record or record.get('analysis_source_sha256') != source_hash or record.get('bootstrap_requested') != args.bootstrap or record['raw_sha256'] != raw_hash:
            record = analyze_surface(path, metadata, args.bootstrap)
            record.update(analysis_source_sha256=source_hash, bootstrap_requested=args.bootstrap)
            write(target, record)
            print('ANALYZED', task['id'], record['headline'].get('p'), flush=True)
        rows.append(record)
    primary = [r for r in rows if r['tag'] == 'primary']
    pairs = {p['id']: p for p in plan['pairs']}
    groups = {'all_features': [r for r in primary if r['kind'] != 'control'],
              'paper_features': [r for r in primary if r['kind'] == 'paper'],
              'novel_features': [r for r in primary if r['kind'] == 'novel'],
              'stratified_only': [r for r in primary if 'stratified_sample' in pairs[r['pair_id']]['reasons']],
              'fixed_challenges': [r for r in primary if 'fixed_challenge' in pairs[r['pair_id']]['reasons']]}
    for r in primary:
        if r['kind'] == 'control':
            groups.setdefault(r['stratum'], []).append(r)
    grouped = {name: {**summarize(values), 'direction_bootstrap': direction_bootstrap(values)} for name, values in groups.items()}
    strata = {label: summarize([r for r in primary if r['kind'] != 'control' and r['stratum'] == label])
              for label in sorted({r['stratum'] for r in primary if r['kind'] != 'control'})}
    unflagged = {name: summarize([r for r in values if not r['headline'].get('flags')])
                 for name, values in groups.items()}
    # Same-pair reference comparisons use only the paper's own eligible Gemma CSV.
    paired_reference = []
    reference_path = args.reference_dir / 'results/conic_comparison_gemma_L2.csv'
    if reference_path.exists():
        reference = {}
        for r in csv.DictReader(reference_path.open()):
            if r['family'] == 'Contrastive':
                name = r['file'].removeprefix('sweep2d_gemma_L2_').removesuffix('_fineweb_60deg.pkl')
                reference[frozenset(name.split('__'))] = float(r['p'])
        for r in primary:
            key = frozenset([r['a'], r['b']])
            if r['kind'] == 'paper' and key in reference:
                paired_reference.append({'pair_id': r['pair_id'], 'a': r['a'], 'b': r['b'],
                                         'gemma_cached_p': reference[key], 'qwen4b_p': r['headline']['p']})
    eligibility = Counter()
    for a, b in itertools.combinations(plan['feature_metadata'], 2):
        eligibility['×'.join(sorted([plan['feature_metadata'][a]['family'], plan['feature_metadata'][b]['family']]))] += 1
    rejections = Counter(r['stratum'] for r in plan['rejected_feature_pairs'])
    comparison = []
    primary_by_id = {r['pair_id']: r for r in primary}
    for row in rows:
        if row['tag'] != 'primary' and row['pair_id'] in primary_by_id:
            p = primary_by_id[row['pair_id']]['headline']['p']
            baseline_anchor_bank = primary_by_id[row['pair_id']]['anchor_bank']
            if row['tag'].startswith('signs'):
                data = np.load(run / 'surfaces' / (row['pair_id'] + '_primary.npz'))
                indices = [baseline_anchor_bank.index(i) for i in row['anchor_bank']]
                matched_grid = np.median(data['l2'][indices], axis=0)
                p = fit(data['angles_deg'], matched_grid, threshold(matched_grid))['p']
                baseline_anchor_bank = row['anchor_bank']
            q = row['headline']['p']
            comparison.append({'task': row['id'], 'a': row['a'], 'b': row['b'], 'tag': row['tag'],
                               'matched_primary_p': p, 'baseline_anchor_bank': baseline_anchor_bank,
                               'diagnostic_p': q, 'difference': q - p if p is not None and q is not None else None})
    # Anchor resamples are shared across every primary pair (same indices/seed).
    # These intervals describe anchor sensitivity of this fixed cohort, not independent features or models.
    anchor_contrasts = {}
    controls = [r for r in primary if r['kind'] == 'control']
    for label in ['all_features', 'paper_features', 'novel_features', 'stratified_only']:
        features = groups[label]
        if not features or not controls or not args.bootstrap:
            continue
        draws = [r.get('anchor_bootstrap', {}).get('p_draws', []) for r in features + controls]
        if not all(len(d) == args.bootstrap for d in draws):
            continue
        feature_draws = np.array(draws[:len(features)])
        control_draws = np.array(draws[len(features):])
        delta = np.median(feature_draws, axis=0) - np.median(control_draws, axis=0)
        anchor_contrasts[label] = {'median_p_difference_vs_pooled_controls': float(np.median(finite_ps(features)) - np.median(finite_ps(controls))),
                                  'paired_anchor_interval_90': np.percentile(delta, [5, 95]).tolist(),
                                  'scope': 'Fixed feature/control cohort, shared anchor resamples; control families pooled descriptively.'}
    summary = {'complete': (run / 'COMPLETE.json').exists(), 'completed_surfaces': len(rows), 'planned_surfaces': len(plan['tasks']),
               'primary_completed': len(primary), 'groups': grouped, 'paired_gemma_reference': paired_reference,
               'feature_strata': strata, 'unflagged_sensitivity_only': unflagged,
               'fixed_cohort_anchor_contrasts': anchor_contrasts,
               'eligibility': {k: {'possible': n, 'eligible': n - rejections[k]} for k, n in eligibility.items()},
               'diagnostics': comparison, 'source_sha256': source_hash,
               'scope': 'One new model and a prospective stress cohort; feature coverage is not a random sample of all latent features. Controls and feature pairs are few and share directions.'}
    write(out / 'summary.json', summary)
    with (out / 'primary-results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'a', 'b', 'kind', 'stratum', 'reasons', 'cosine', 'p',
                                                    'radial_residual', 'n_contour_points', 'flags', 'anchor_lo90', 'anchor_hi90'])
        writer.writeheader()
        for r in primary:
            interval = r.get('anchor_bootstrap', {}).get('interval_90') or [None, None]
            writer.writerow({'id': r['pair_id'], 'a': r['a'], 'b': r['b'], 'kind': r['kind'], 'stratum': r['stratum'],
                             'reasons': ';'.join(pairs[r['pair_id']]['reasons']), 'cosine': pairs[r['pair_id']]['cosine'],
                             'p': r['headline']['p'], 'radial_residual': r['headline']['mean_radial_frac'],
                             'n_contour_points': r['headline']['n_pts'], 'flags': ';'.join(r['headline'].get('flags', [])),
                             'anchor_lo90': interval[0], 'anchor_hi90': interval[1]})
    # Summary distribution: every finite fit is included; flagged fits are hollow.
    labels = [('paper_features', 'Paper\nfeatures'), ('novel_features', 'New\nfeatures'),
              ('isotropic_random', 'Isotropic\nrandom'), ('random_activation_difference_mean30', 'Random\nactivation diff.'),
              ('same_prompt_random_relabelling', 'Same prompts,\nrandom labels')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={'width_ratios': [1.3, 1.3, 1]})
    for i, (key, label) in enumerate(labels):
        values = groups.get(key, [])
        valid = [r for r in values if r['headline']['p'] is not None]
        jitter = np.linspace(-.18, .18, len(valid))
        color = '#367d68' if i < 2 else '#777d8a'
        for offset, row in zip(jitter, valid):
            for ax in axes[:2]:
                ax.scatter(i + offset, row['headline']['p'], s=40, edgecolor=color,
                           facecolor='none' if row['headline'].get('flags') else color, zorder=3)
        ps = finite_ps(values)
        if len(ps):
            for ax in axes[:2]:
                ax.plot([i - .28, i + .28], [np.median(ps)] * 2, color=color, linewidth=2)
    for ax in axes[:2]:
        ax.axhline(2, color='#b66b36', linestyle='--', linewidth=1.3)
        ax.set(xticks=range(len(labels)), xticklabels=[x[1] for x in labels], ylabel='Fitted superellipse p')
        ax.tick_params(axis='x', labelsize=8)
    axes[0].set_title('All completed primary fits')
    axes[1].set(ylim=(1.5, 3.5), title='Same points, central range enlarged')
    finite_reference = [r for r in paired_reference if r['qwen4b_p'] is not None]
    if finite_reference:
        x = [r['gemma_cached_p'] for r in finite_reference]; y = [r['qwen4b_p'] for r in finite_reference]
        axes[2].scatter(x, y, color='#367d9c', alpha=.8)
        extent = [min(.5, min(x+y)), max(3., max(x+y))]
        axes[2].plot(extent, extent, color='#777777', linestyle=':')
        axes[2].axhline(2, color='#b66b36', linestyle='--', linewidth=1)
        axes[2].axvline(2, color='#b66b36', linestyle='--', linewidth=1)
    axes[2].set(xlabel='Gemma cached p, same eligible named pair', ylabel='Fresh Qwen3.5-4B p', title='Same named pairs, adapted setup')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.16); ax.set_axisbelow(True)
    fig.suptitle(f'Qwen3.5-4B stress test — {len(rows)}/{len(plan["tasks"])} surfaces complete', fontsize=14)
    fig.text(.5, -.02, 'Eight fixed DCLM five-token anchors, block 2 → 30. Hollow points: fit flags. The cohort includes fixed challenges and coverage additions.', ha='center', fontsize=9)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'primary-results.{extension}', dpi=170, bbox_inches='tight', facecolor='white')
    print('SUMMARY', json.dumps(clean({k: {kk: v for kk, v in values.items() if kk != 'direction_bootstrap'} for k, values in grouped.items()})), flush=True)


if __name__ == '__main__':
    main()
