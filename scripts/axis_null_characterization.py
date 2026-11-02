"""Predict primary 2-D response geometry from the saved single-axis norms."""
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_toy_cis as F

ROOT = Path(__file__).resolve().parents[1]


def main(family):
    source = ROOT / 'reports' / ('toy-characterization-v1' if family == 'orthogonal' else 'toy-characterization-coupled-v1')
    output = ROOT / 'reports/toy-characterization-axis-null-v1'
    output.mkdir(exist_ok=True)
    assert not (output / (family + '.json')).exists()
    plan = json.loads((source / 'analysis-plan.json').read_text())
    files = sorted((source / 'analysis').glob('*/*.json'))
    assert len(files) == len(plan['networks'])
    records = []; max_control_error = 0.
    for path in files:
        full = json.loads(path.read_text()); meta = full['network']
        with np.load(path.with_suffix('.npz')) as raw:
            for index, row in enumerate(full['geometry']):
                if row['kind'] != 'feature_axes' or row['anchor'] != 0 or row['sign'] != [1, 1]: continue
                grid = raw['response'][index]
                predicted = np.hypot(grid[:, :, 0, None], grid[:, None, 0, :])
                absolute = float(np.abs(grid - predicted).max())
                if row['map'] == 'separable':
                    max_control_error = max(max_control_error, absolute)
                    assert absolute < 1e-8
                median = np.median(predicted, axis=0)
                fits = {str(f): F.fit(median, f)[0] for f in [.25, .5, .75]}
                original_p = row['fits']['0.5']['p']; predicted_p = fits['0.5']['p']
                residual = float(np.mean(np.abs(np.median(grid, axis=0) - median)) / max(np.max(np.median(grid, axis=0)), 1e-12))
                records.append({'network': meta, 'pair': row['pair'], 'map': row['map'],
                                'original_fits': row['fits'], 'axis_null_fits': fits,
                                'per_background': [F.fit(g, .5)[0] for g in predicted],
                                'mean_response_residual_fraction': residual,
                                'max_abs_response_difference': absolute,
                                'p_minus_axis_null': original_p - predicted_p if original_p is not None and predicted_p is not None else None,
                                'source_raw_sha256': full['raw_sha256']})
        print('AXIS_NULL', family, path.stem, flush=True)
    F.write(output / (family + '.json'), {'records': records, 'family': family,
             'separable_max_abs_grid_identity_error': max_control_error,
             'source_sha256': F.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'protocol_sha256': F.hashlib.sha256((ROOT / 'configs/toy-characterization-axis-null-v1.json').read_bytes()).hexdigest(),
             'all_control_identity_assertions_passed': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['orthogonal', 'coupled'], required=True)
    main(parser.parse_args().family)
