"""Reproduce the pinned paper Figure 2 from its single cached sweep, using git."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a9355d4001500708a40a255bacfe559e6dc269c0'
CACHE = 'results/sweeps_2d/sweep2d_gemma_L2_Wealth__Gender_fineweb_60deg.pkl'
CACHE_GIT_BLOB = 'a3c0ac5524dd7ec841a3a899d74dddce925ae998'
SOURCES = ['LICENSE', 'scripts/lib/superellipse.py', 'scripts/plotting/plot_fig3_boundary.py']


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'work/fsec-reference')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/paper')
    args = parser.parse_args()
    ref = args.reference_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        run('git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
            'https://github.com/FranciscoHS/fsec-paper.git', str(ref))
        run('git', 'sparse-checkout', 'set', '--no-cone',
            *('/' + path for path in SOURCES + [CACHE]), cwd=ref)
        run('git', 'fetch', '--depth', '1', 'origin', REVISION, cwd=ref)
        run('git', 'checkout', '--detach', REVISION, cwd=ref)
    assert run('git', 'rev-parse', 'HEAD', cwd=ref) == REVISION
    run('git', 'diff', '--exit-code', 'HEAD', '--', *SOURCES, CACHE, cwd=ref)
    assert run('git', 'hash-object', CACHE, cwd=ref) == CACHE_GIT_BLOB
    # Use only the original cache whose pinned Git object was checked above.
    with (ref / CACHE).open('rb') as stream:
        data = pickle.load(stream)
    for path in SOURCES:
        destination = ROOT / 'vendor/fsec' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref / path, destination)
    command = [sys.executable, 'scripts/plotting/plot_fig3_boundary.py', '--target', 'gemma',
               '--layer', '2', '--d1', 'Wealth', '--d2', 'Gender', '--exact']
    result = subprocess.run(command, cwd=ref, check=True, text=True, capture_output=True)
    (out / 'console.log').write_text(result.stdout + result.stderr)
    print(result.stdout, end='')
    stem = 'fig3_boundary_Wealth_Gender_gemma_L2_exact'
    for extension in ['png', 'pdf']:
        shutil.copyfile(ref / 'results/figures' / f'{stem}.{extension}',
                        out / f'paper-figure-2.{extension}')

    sys.path.insert(0, str(ref))
    from scripts.lib.superellipse import axis_intercept, extract_contour, fit_superellipse
    angles = data['angles_deg']
    grid = np.median(data['l2'], axis=0)
    threshold = 0.5 * min(float(grid[:, 0].max()), float(grid[0, :].max()))
    a1 = axis_intercept(angles, grid[:, 0], threshold)
    a2 = axis_intercept(angles, grid[0, :], threshold)
    raw = extract_contour(angles, grid, threshold)
    radius = np.hypot(raw[:, 0], raw[:, 1])
    safe_radius = np.where(radius > 1e-12, radius, 1.)
    x = np.sin(np.deg2rad(radius)) * raw[:, 0] / safe_radius / np.sin(np.deg2rad(a1))
    y = np.sin(np.deg2rad(radius)) * raw[:, 1] / safe_radius / np.sin(np.deg2rad(a2))
    fit = fit_superellipse(x, y)
    assert np.isfinite(fit['p']) and abs(fit['p'] - 2.40) < 0.02
    assert abs(fit['mean_radial_frac'] - 0.012) < 0.003
    np.savetxt(out / 'contour.csv', np.column_stack([raw, x, y]), delimiter=',',
               header='alpha1_deg,alpha2_deg,normalized_x,normalized_y', comments='')
    np.savez_compressed(out / 'cached-responses.npz', angles_deg=angles,
                        l2_per_anchor=data['l2'], l2_median=grid)
    summary = {
        'experiment': 'Cached paper Figure 2 reproduction; not a new model experiment',
        'repository': 'https://github.com/FranciscoHS/fsec-paper', 'revision': REVISION,
        'cache_path': CACHE, 'cache_git_blob': CACHE_GIT_BLOB,
        'cache_sha256': sha256(ref / CACHE), 'cache_bytes': (ref / CACHE).stat().st_size,
        'source_sha256': {path: sha256(ref / path) for path in SOURCES},
        'command': command, 'packages': {name: importlib.metadata.version(name)
                                      for name in ['numpy', 'scipy', 'matplotlib']},
        'response_shape': list(data['l2'].shape), 'threshold_l2': threshold,
        'axis_intercepts_deg': {'Wealth': a1, 'Gender': a2}, 'fit': fit,
        'paper_comparison': {'reported_p_rounded': 2.40, 'matches_to_2_decimals': round(fit['p'], 2) == 2.40},
        'normalization': 'exact sine geodesic; threshold on median-over-anchor grid',
    }
    (out / 'reproduction.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Verified and saved reproduction to {out}')


if __name__ == '__main__':
    main()
