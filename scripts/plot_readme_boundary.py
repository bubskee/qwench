"""Show measured round/squarer toy contours from an explicitly selected slice.

Uses the same first competence-conditioned pair in two seed-0 L2 networks.
No new inference, fitting, reflected quadrants, or p-based pair search.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/toy-scalar-comparison-v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'work/reproductions/readme-boundary.png')
    args = parser.parse_args()
    output = args.out.resolve()
    receipt = output.with_suffix('.json')
    if output.exists() or receipt.exists():
        parser.error('Output exists; choose another --out.')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'vendor/fsec/scripts/lib/superellipse.py'
    spec = importlib.util.spec_from_file_location('fsec_fit', source)
    fitter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fitter)
    plan = json.loads((REPORT / 'geometry-plan.json').read_text())
    evaluations = json.loads((REPORT / 'eval-100000.json').read_text())['networks']
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharex=True, sharey=True)
    records = []
    common_pair = None
    for ax, ident, name, color in zip(axes,
            ['relu50_L2_s0', 'relu2_50_L2_s0'],
            ['ReLU: round', 'Squared ReLU: squarer'], ['#377bad', '#9364a4']):
        network = next(row for row in plan['networks'] if row['id'] == ident)
        pair = network['conditioned_pairs'][0]
        if common_pair is None:
            common_pair = pair
        assert pair == common_pair, 'This comparison must use the same feature pair.'
        evaluation = next(row for row in evaluations if row['id'] == ident)
        assert evaluation['competent_features'] == 50
        assert abs(evaluation['mean_neuron_feature_participation'] - 1) < 1e-3
        raw_path = REPORT / 'surfaces' / (ident + '.npz')
        meta = json.loads(raw_path.with_suffix('.json').read_text())
        analysis_path = REPORT / 'analysis' / (ident + '.json')
        analysis = json.loads(analysis_path.read_text())
        index = next(i for i, row in enumerate(meta['rows']) if row['features'] == pair
                     and row['kind'] == 'encoder_columns' and row['anchor_value'] == 0
                     and row['sign'] == [1, 1])
        fit = analysis['rows'][index]['fits']['0.5']
        assert not fit['flags']
        with np.load(raw_path) as data:
            median = np.median(data['l2'][index], axis=0)
        axis = np.linspace(0, 1.5, median.shape[0])
        points = fitter.extract_contour(axis, median, fit['threshold'])
        points /= np.asarray(fit['intercepts_feature_units'])
        points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
        polygon = np.vstack([[0, 0], points, [0, 0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=.10)
        ax.plot(points[:, 0], points[:, 1], color=color, lw=3)
        ax.text(.34, .34, 'Below the\nchange cutoff', ha='center', va='center',
                fontsize=12, color='#444444', linespacing=1.4)
        ax.set(title=f'{name}\np = {fit["p"]:.2f}', xlabel='Turn up feature A',
               xlim=(0, 1.09), ylim=(0, 1.09), xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.12)
        records.append({'id': ident, 'features': pair, 'anchor_value': 0,
                        'sign': [1, 1], 'kind': 'encoder_columns', 'fit': fit,
                        'raw_sha256': sha(raw_path), 'analysis_sha256': sha(analysis_path)})
    axes[0].set_ylabel('Turn up feature B')
    fig.suptitle('Different shapes. Both models use roughly one feature per neuron.',
                 fontsize=13, y=.98)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.text(.5, .025, 'Measured positive-direction slices; axes rescaled to their one-feature cutoffs.',
             ha='center', fontsize=10, color='#555555')
    fig.savefig(output, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    receipt.write_text(json.dumps({'selection': 'First competence-conditioned pair in each seed-0 L2 network; same pair, no search over p.',
                       'background_aggregation': 'median of four frozen backgrounds',
                       'source_sha256': sha(Path(__file__)), 'models': records}, indent=2) + '\n')
    print(f'Saved {output} and {receipt}')


if __name__ == '__main__':
    main()
