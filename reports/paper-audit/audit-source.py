"""Audit cached FSEC results with pinned original scripts and sampled raw-grid refits.

Requires numpy/scipy/matplotlib and CPU PyTorch for tensor-containing pickles.
No LLM weights or inference. Downloads selected data through a sparse git checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REV = 'a9355d4001500708a40a255bacfe559e6dc269c0'
EXCLUDED = {'Formal', 'HonestyShort', 'TensePresent'}
CELL = ('1.0xT', 60.0)


def git(ref, *args):
    return subprocess.check_output(['git', '-C', str(ref), *args], text=True).strip()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    with path.open('rb') as f:
        return pickle.load(f)


def stats(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'n': 0}
    return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'q25': float(np.percentile(a, 25)), 'q75': float(np.percentile(a, 75)),
            'min': float(a.min()), 'max': float(a.max()), 'fraction_gt_2': float((a > 2).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-dir', type=Path, default=ROOT / 'work/fsec-reference')
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/paper-audit')
    args = ap.parse_args()
    ref, out = args.reference_dir.resolve(), args.out.resolve()
    assert git(ref, 'rev-parse', 'HEAD') == REV
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    torch.set_num_threads(6)
    patterns = ['/scripts/plotting', '/scripts/lib', '/results/fits', '/results/directions', '/data', '/docs',
                '/results/pos_pairs_gemma_L2.csv', '/results/conic_comparison_gemma_L2.csv',
                '/jlens/README.md', '/jlens/plot_beeswarm.py', '/jlens/refit_protocol.py',
                '/jlens/data/fits_contrastive_thrpair_exact.pkl', '/jlens/data/fits_jlens_en_concrete_thrpair_exact.pkl',
                '/jlens/data/fits_jlens_zh100_thrpair_exact.pkl', '/toy_model/scripts', '/toy_model/mft_denoising',
                '/toy_model/runs/misalignment_sweep_H1024/lp_vs_angle_no_feature_orth.npz',
                '/toy_model/runs/misalignment_sweep_H1024/phase2_d8192_H1024.npz']
    git(ref, 'sparse-checkout', 'add', *patterns)
    os.chdir(ref)
    sys.path.insert(0, str(ref))
    from scripts.plotting import plot_robustness_beeswarm as P
    from scripts.plotting import plot_beeswarm_direction_types as F
    from scripts.plotting import plot_misalignment_llm as M
    from scripts.lib.superellipse import fit_p_one_pass
    P.USE_THRPAIR = True
    P.USE_EXACT = True
    all_sweeps = set(git(ref, 'ls-tree', '-r', '--name-only', 'HEAD', 'results/sweeps_2d').splitlines())
    random_sweeps, seen = [], set()
    prefix = 'results/sweeps_2d/sweep2d_gemma_L2_'
    for path in sorted(all_sweeps):
        if not (path.startswith(prefix) and path.endswith('_fineweb_60deg_dirrandom.pkl')):
            continue
        first = path[len(prefix):].split('__')[0]
        if first not in seen:
            random_sweeps.append(path)
            seen.add(first)
        if len(random_sweeps) == 10:
            break
    assert len(random_sweeps) == 10
    selected, families, source_fits = [], {}, {}
    for label, suffix, dir_suffix, _ in F.FAMILIES:
        fit_path = ref / f'results/fits/fits_gemma_L2{suffix}_thrpair_exact.pkl'
        full = load(fit_path)
        source_fits[label] = full
        values = F._load_fit_pair_dict(str(fit_path))
        values = F._filter_family_overlap(values, 'gemma', 2, dir_suffix, .1)
        if dir_suffix is None:
            values = P._filter_exclude(values, EXCLUDED)
        values = {key: val for key, val in values.items() if np.isfinite(val)}
        families[label] = stats(values.values())
        pairs = sorted(values, key=lambda pair: tuple(sorted(pair)))
        # Three fixed label-order positions; never select by the observed exponent.
        for index in sorted({0, len(pairs) // 2, len(pairs) - 1}):
            pair = pairs[index]
            a, b = sorted(pair)
            possible = [f'{prefix}{x}__{y}_fineweb_60deg{suffix}.pkl' for x, y in [(a, b), (b, a)]]
            matches = [p for p in possible if p in all_sweeps]
            assert len(matches) == 1, (label, pair, matches)
            selected.append({'family': label, 'pair': [a, b], 'path': matches[0]})
    git(ref, 'sparse-checkout', 'add', *('/' + p for p in random_sweeps + [r['path'] for r in selected]))
    raw_refits = []
    for selection in selected:
        sweep = load(ref / selection['path'])
        grid = np.median(sweep['l2'], axis=0)
        threshold = .5 * min(float(grid[:, 0].max()), float(grid[0, :].max()))
        fit = fit_p_one_pass(sweep['angles_deg'], grid, threshold, 60., exact_geodesic=True)
        pair = frozenset(selection['pair'])
        cached = next(value[CELL] for key, value in source_fits[selection['family']].items() if frozenset(key) == pair)
        cached_point = float(cached['p_point'])
        error = abs(float(fit['p']) - cached_point)
        raw_refits.append({**selection, 'threshold': threshold, 'refit_p': float(fit['p']),
                           'cached_p_point': cached_point, 'absolute_p_error': error,
                           'mean_radial_frac': fit['mean_radial_frac'], 'n_points': fit['n_pts']})
        print(f'Raw refit {selection["family"]}: {selection["pair"]}, error={error:.3g}', flush=True)
    dump(out / 'raw-refits.json', raw_refits)
    # Appendix A, independently recomputed from cached unit vectors.
    directions = load(ref / 'results/directions/dirs_gemma_L2.pkl')['directions']
    names = sorted(set(directions) - EXCLUDED)
    assert len(names) == 33
    matrix = np.stack([directions[n].cpu().numpy().astype(float) for n in names])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    overlap = np.abs(matrix @ matrix.T)
    mask = np.triu(np.ones(overlap.shape, dtype=bool), k=1)
    appendix_a = {'directions': len(names), 'all_pairs': int(mask.sum()),
                  'kept_pairs_abs_cos_lt_0p1': int(((overlap < .1) & mask).sum()),
                  'dropped_pairs': int(((overlap >= .1) & mask).sum())}
    # Ablation statistics use exactly the original script's filtering order.
    ablations = {}
    loaders = [('model', lambda: P.col_model(2, max_overlap=.1)),
               ('perturbation_layer', lambda: P.col_perturb_layer('gemma', max_overlap=.1)),
               ('measurement_layer', lambda: P.col_measure_layer('gemma', 2, max_overlap=.1)),
               ('metric', lambda: P.col_metric('gemma', 2, max_overlap=.1)),
               ('threshold', lambda: P.col_threshold('gemma', 2, max_overlap=.1)),
               ('method', lambda: P.col_method('gemma', 2, max_overlap=.1)),
               ('anchor_source', lambda: P.col_anchor_source('gemma', 2, max_overlap=.1)),
               ('token_position', lambda: P.col_token_position('gemma', 2, max_overlap=.1))]
    for label, loader in loaders:
        groups = P._apply_filters(loader(), 'gemma', 2, .1, EXCLUDED)
        ablations[label] = {key: stats(values.values()) for key, values in groups.items()}
    # Residual distributions, using original loader functions with residual as statistic.
    from scripts.plotting import plot_residuals_beeswarm as R
    original = P.median_p_for_pair
    P.median_p_for_pair = R.median_residual_for_pair
    residuals = {}
    for label, loader in loaders:
        groups = R.col_threshold_residual('gemma', 2, max_overlap=.1) if label == 'threshold' else loader()
        groups = P._apply_filters(groups, 'gemma', 2, .1, EXCLUDED)
        residuals[label] = {key: stats(values.values()) for key, values in groups.items()}
    for label, suffix, dir_suffix, _ in F.FAMILIES:
        values = P._load_fit_pair_dict(f'results/fits/fits_gemma_L2{suffix}_thrpair_exact.pkl')
        values = F._filter_family_overlap(values, 'gemma', 2, dir_suffix, .1)
        if dir_suffix is None:
            values = P._filter_exclude(values, EXCLUDED)
        residuals.setdefault('direction_family', {})[label] = stats(values.values())
    P.median_p_for_pair = original
    rotations = {str(theta): stats(values) for theta, values in M._collect('gemma', 2, '_thrpair_exact').items()}
    # Later repository controls: pooled shuffled weights and matched trained subset.
    shuffled, shuffled_filtered, tier_pairs = [], [], set()
    for seed in range(5):
        path = ref / f'results/fits/fits_gemma_L2_shufseed{seed}_thrpair_exact.pkl'
        values = P._load_fit_pair_dict(str(path))
        shuffled.extend(values.values())
        tier_pairs.update(values)
        filtered = F._filter_family_overlap(values, 'gemma', 2, f'_shufseed{seed}', .1)
        shuffled_filtered.extend(filtered.values())
    trained = P.load_default_fits('gemma', 2)
    shuffled_summary = {'unfiltered': stats(shuffled), 'overlap_filtered': stats(shuffled_filtered),
                        'same_label_pairs_in_trained_model': stats(v for key, v in trained.items() if key in tier_pairs)}
    # Later J-lens addition: angle windows intentionally remain unequal.
    jlens = {}
    for label, name, window in [('contrastive', 'fits_contrastive_thrpair_exact.pkl', 60.),
                                ('J_Chinese', 'fits_jlens_zh100_thrpair_exact.pkl', 20.),
                                ('J_English', 'fits_jlens_en_concrete_thrpair_exact.pkl', 20.)]:
        full = load(ref / 'jlens/data' / name)
        values = [v[('1.0xT', window)]['p_median'] for v in full.values() if ('1.0xT', window) in v]
        jlens[label] = {**stats(values), 'angle_window_deg': window}
    toy_path = ref / 'toy_model/runs/misalignment_sweep_H1024/lp_vs_angle_no_feature_orth.npz'
    toy_cache = np.load(toy_path, allow_pickle=False)
    toy = {}
    for d in toy_cache['ds']:
        values = toy_cache[f'ns_d{int(d)}']
        if values.ndim == 3:
            values = values[0]
        toy[str(int(d))] = {'H': int(toy_cache['H']), 'per_angle': [stats(row) for row in values]}

    jobs = [
        ('figure1_response', ['scripts/plotting/plot_combo_sweep.py', '--target', 'gemma', '--layer', '2', '--d1', 'Wealth', '--d2', 'Gender', '--max_angle', '60', '--out_tag', 'thrpair']),
        ('figure4_families', ['scripts/plotting/plot_beeswarm_direction_types.py', '--target', 'gemma', '--layer', '2', '--max_overlap', '.10', '--exclude_dirs', 'HonestyShort,TensePresent,Formal', '--use_thrpair', '--exact']),
        ('figure5_rotation', ['scripts/plotting/plot_misalignment_llm.py', '--target', 'gemma', '--layer', '2', '--exact']),
        ('figure6_ablations', ['scripts/plotting/plot_robustness_beeswarm.py', '--target', 'gemma', '--layer', '2', '--max_overlap', '.10', '--exclude_dirs', 'Formal,HonestyShort,TensePresent', '--use_thrpair', '--exact']),
        ('appendixA_overlap', ['scripts/plotting/build_directions_appendix.py']),
        ('appendixB_residuals', ['scripts/plotting/plot_residuals_beeswarm.py']),
        ('figure7_toy_rotation_cached', ['toy_model/scripts/plot_misalignment_lp.py', '--npz', str(toy_path), '--out', str(out / 'figures/figure7_toy_rotation_cached.png')]),
        ('jlens_later_addition', ['jlens/plot_beeswarm.py', '--out', str(out / 'figures/jlens_later_addition')]),
    ]
    job_records = []
    for label, argv in jobs:
        before = {p: p.stat().st_mtime_ns for p in (ref / 'results/figures').glob('*')}
        start = time.perf_counter()
        result = subprocess.run([sys.executable, *argv], cwd=ref, text=True, capture_output=True)
        (out / 'logs' / f'{label}.log').write_text(result.stdout + result.stderr)
        artifacts = []
        for path in (ref / 'results/figures').glob('*'):
            if path.suffix in ['.png', '.pdf', '.tex'] and before.get(path) != path.stat().st_mtime_ns:
                destination = out / 'figures' / f'{label}{path.suffix}'
                shutil.copyfile(path, destination)
                artifacts.append(str(destination.relative_to(out)))
        job_records.append({'name': label, 'argv': argv, 'returncode': result.returncode,
                            'seconds': time.perf_counter() - start, 'copied_artifacts': artifacts})
        dump(out / 'render-jobs.json', job_records)
        print(f'Render {label}: exit={result.returncode}', flush=True)
        if result.returncode:
            print(result.stderr[-2000:], flush=True)
    summary = {'scope': 'Cached-data audit; 18 sampled raw-grid refits, original-script rerenders; no fresh LLM forwards.',
               'revision': REV, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'families': families, 'appendix_a': appendix_a, 'ablations': ablations,
               'residuals': residuals, 'rotation': rotations, 'toy_cached_rotation': toy,
               'shuffled_weights_later_addition': shuffled_summary, 'jlens_later_addition': jlens,
               'raw_refit_count': len(raw_refits), 'max_raw_refit_p_error': max(r['absolute_p_error'] for r in raw_refits),
               'render_jobs': job_records,
               'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'matplotlib', 'torch']}}
    dump(out / 'summary.json', summary)
    inputs = []
    for directory in ['results/fits', 'results/directions', 'scripts/plotting', 'scripts/lib', 'data', 'docs', 'jlens', 'toy_model/scripts', 'toy_model/mft_denoising', 'toy_model/runs']:
        for path in (ref / directory).rglob('*'):
            if path.is_file() and path.suffix in ['.pkl', '.npz', '.py', '.md']:
                inputs.append(path)
    inputs += [ref / 'results/pos_pairs_gemma_L2.csv', ref / 'results/conic_comparison_gemma_L2.csv']
    inputs += [ref / r['path'] for r in selected] + [ref / p for p in random_sweeps]
    with (out / 'input-manifest.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'bytes', 'sha256'])
        for path in sorted(set(inputs)):
            writer.writerow([str(path.relative_to(ref)), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()])
    print(f'Saved audit to {out}', flush=True)


if __name__ == '__main__':
    main()
