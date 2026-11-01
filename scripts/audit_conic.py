"""Recompute selected superellipse-versus-tilted-ellipse fits from cached raw grids."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a9355d4001500708a40a255bacfe559e6dc269c0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'work/fsec-reference')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/paper-audit')
    args = parser.parse_args()
    ref, out = args.reference_dir.resolve(), args.out.resolve()
    revision = subprocess.check_output(['git', '-C', str(ref), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == REVISION
    subprocess.run(['git', '-C', str(ref), 'sparse-checkout', 'add', '/scripts/fit_conic_comparison.py', '/results/conic_comparison_gemma_L2.csv'], check=True)
    sys.path.insert(0, str(ref))
    from scripts.fit_conic_comparison import normalized_points, fit_tilted_ellipse
    from scripts.lib.superellipse import fit_superellipse
    path = ref / 'results/conic_comparison_gemma_L2.csv'
    rows = list(csv.DictReader(path.open()))
    cached = {(r['family'], r['file']): r for r in rows}
    refits = []
    for selection in json.loads((out / 'raw-refits.json').read_text()):
        points = normalized_points(ref / selection['path'])
        assert points is not None
        superellipse = fit_superellipse(*points)
        delta, residual = fit_tilted_ellipse(*points)
        original = cached[(selection['family'], Path(selection['path']).name)]
        actual = {'p': superellipse['p'], 'resid_p': superellipse['mean_radial_frac'], 'delta': delta, 'resid_delta': residual}
        errors = {key: abs(float(value) - float(original[key])) for key, value in actual.items()}
        refits.append({**selection, 'conic_parameters': actual, 'cached_csv_absolute_errors': errors})
    families = {}
    for family in dict.fromkeys(row['family'] for row in rows):
        selected = [row for row in rows if row['family'] == family]
        a = np.array([float(row['resid_p']) for row in selected])
        b = np.array([float(row['resid_delta']) for row in selected])
        families[family] = {'n': len(selected), 'median_superellipse_residual': float(np.median(a)),
                            'median_tilted_ellipse_residual': float(np.median(b)),
                            'tilted_ellipse_wins': int((b < a).sum()), 'superellipse_wins': int((a < b).sum()),
                            'non_positive_definite_conics': sum(abs(float(row['delta'])) >= 1 for row in selected)}
    result = {'scope': 'Later repository conic comparison, not present in the inspected arXiv v1 appendices.',
              'revision': revision, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ref / 'scripts/fit_conic_comparison.py').read_bytes()).hexdigest(),
              'families': families, 'sampled_raw_refits': refits,
              'max_cached_csv_error': max(value for row in refits for value in row['cached_csv_absolute_errors'].values()),
              'limitations': ['Fits are evaluated on the same contour points used for fitting.',
                              'Pairs share directions; a naive pair-level sign test is not independent evidence.',
                              'Relative goodness of fit does not identify an error-correction mechanism.']}
    (out / 'conic-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Conic comparison: sampled maximum absolute error', result['max_cached_csv_error'])


if __name__ == '__main__':
    main()
