"""Independent derivative and separability checks for the tiny recovery probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import squared_recovery as R


def main(out):
    with np.load(out / 'models.npz') as weights, np.load(out / 'inputs.npz') as data:
        x = R.tensor(data['clean_x'][:8].copy())
        direction = R.tensor(data['standard_noise'][0, :8].copy())
        pair = data['pair'].tolist()
        supported = data['supported'].copy()
        checks = []
        models = {name: {role: R.tensor(weights[f'{name}_{role}'].copy()) for role in ('gate', 'out')}
                  for name in ('trained', 'independent')}
    for name, model in models.items():
        _, automatic = torch.autograd.functional.jvp(lambda value: R.forward(value, model), x, direction)
        z = x @ model['gate'].T
        explicit = (2 * z.relu() * (direction @ model['gate'].T)) @ model['out'].T
        error = float((automatic - explicit).abs().max())
        assert error < 1e-10
        checks.append({'model': name, 'autograd_jacobian_vector_max_error': error})
    control = models['independent']
    eye = torch.eye(100, dtype=torch.float64)
    isolated = R.forward(eye, control)
    off = isolated - torch.diag(isolated.diagonal())
    assert float(off.abs().max()) == 0
    combined = eye[pair].sum(0)[None]
    difference = float((R.forward(combined, control) - isolated[pair].sum(0)).abs().max())
    assert difference < 1e-12
    trained_gains = R.forward(eye, models['trained']).diagonal()[supported]
    assert float((trained_gains - isolated.diagonal()[supported]).abs().max()) < 1e-10
    R.write(out / 'implementation-checks.json', {
        'all_assertions_passed': True, 'derivative_checks': checks,
        'control_off_diagonal_unit_probe_output': float(off.abs().max()),
        'control_pair_additivity_max_error': difference,
        'prepared_sha256': R.sha(out / 'prepared.json'), 'source_sha256': R.sha(__file__)
    })
    print('PASS: exact positive-ray matches, no control cross-talk, oracle derivatives match autograd.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=R.ROOT / 'reports/squared-recovery-v1')
    main(parser.parse_args().out.resolve())
