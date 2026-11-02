"""Secondary projection-matched and bounded-output checks; no new training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import characterization_core as B
import analyze_characterization as A


def project(x, a, task):
    result = x.clone()
    for sign in [-1., 1.]:
        mask = x[:, -1] == sign
        if not mask.any(): continue
        aa = torch.tensor(B.effective_encoder(a, task, sign), dtype=torch.float64)
        projection = aa @ torch.linalg.inv(aa.T @ aa) @ aa.T
        result[mask, :32] = x[mask, :32] @ projection
    return result


def main(family):
    if family == 'orthogonal': C = B; null = A.separable
    else:
        import coupled_characterization as C
        null = C.separable
    config = B.ROOT / 'configs/toy-characterization-diagnostics-v1.json'
    protocol = json.loads(config.read_text())
    output = B.ROOT / 'reports/toy-characterization-diagnostics-v1'
    output.mkdir(exist_ok=True)
    assert not (output / (family + '.json')).exists()
    plan = json.loads((C.OUT / 'analysis-plan.json').read_text())
    records = [row for row in plan['networks'] if row['track'] == 'new' and row['step'] in protocol['checkpoints']]
    with np.load(C.OUT / 'analysis-inputs.npz') as raw: assets = {key: raw[key].copy() for key in raw.files}
    bits = assets['recovery_bits']; context = assets['recovery_context']; encoder = assets['encoder']
    all_rows = []; raw_arrays = {}; checks = []; original_error = 0.
    for meta in records:
        model = A.model_from(meta)
        original = json.loads((C.OUT / 'analysis/new' / (meta['key'] + '.json')).read_text())
        base = C.task_inputs(torch.tensor(bits, dtype=torch.float64), torch.tensor(context, dtype=torch.float64),
                             torch.tensor(encoder), meta['task'])
        for li, sigma in enumerate(C.P['recovery']['noise_std']):
            noisy = base[None].expand(2, *base.shape).clone()
            noisy[..., :32] += sigma * torch.tensor(assets['recovery_noise'])
            x = noisy.reshape(-1, 33); xp = project(x, encoder, meta['task'])
            natural = C.forward(x, model, meta['activation']).numpy().reshape(2, *bits.shape)
            projected = C.forward(xp, model, meta['activation']).numpy().reshape(2, *bits.shape)
            restricted = np.concatenate([null(batch, model, meta, encoder).numpy() for batch in x.split(256)]).reshape(2, *bits.shape)
            maps = {'natural_raw': natural, 'natural_clipped': np.clip(natural, 0, 1),
                    'projected_raw': projected, 'projected_clipped': np.clip(projected, 0, 1),
                    'separable_raw': restricted, 'separable_clipped': np.clip(restricted, 0, 1)}
            for name, prediction in maps.items():
                error = np.square(prediction - bits)
                row = {**meta, 'family': family, 'sigma': sigma, 'map': name}
                for label, mask in [('all', np.ones_like(bits, dtype=bool)), ('active', bits > 0), ('inactive', bits == 0)]:
                    sums = (error * mask).sum(-1).mean(0); counts = mask.sum(-1)
                    row[label + '_mse'] = float(sums.sum() / counts.sum())
                    raw_arrays[meta['key'] + f'_{li}_{name}_{label}_sum'] = sums
                    raw_arrays[meta['key'] + f'_{li}_{name}_{label}_count'] = counts
                all_rows.append(row)
                if name in ['natural_raw', 'separable_raw']:
                    previous = next(row for row in original['recovery'] if row['sigma'] == sigma and row['map'] == ('natural' if name == 'natural_raw' else 'separable'))
                    original_error = max(original_error, max(abs(row[k + '_mse'] - previous[k + '_mse']) for k in ['all', 'active', 'inactive']))
        # Projection preserves every clean-input feature plane; verify a full measured grid.
        pair = C.P['geometry']['fixed_pairs'][0]; axis = np.linspace(0, 1.5, 31)
        aa, bb = np.meshgrid(axis, axis, indexing='ij'); coeff = np.stack([aa.ravel(), bb.ravel()], -1)
        for sign in [-1., 1.]:
            matrix = C.effective_encoder(encoder, meta['task'], sign)
            z = assets['background_bits'][0].copy(); z[pair] = 0
            obs = matrix @ z + coeff @ matrix[:, pair].T
            x = torch.tensor(np.c_[obs, np.full(len(obs), sign)])
            xp = project(x, encoder, meta['task'])
            input_error = float((x - xp).abs().max())
            output_error = float((C.forward(x, model, meta['activation']) - C.forward(xp, model, meta['activation'])).abs().max())
            assert input_error < 1e-10 and output_error < 1e-9
            checks.append({'key': meta['key'], 'context': sign, 'max_input_error': input_error, 'max_output_error': output_error})
        print('DIAGNOSED', family, meta['key'], flush=True)
    assert original_error < 1e-10
    np.savez_compressed(output / (family + '-errors.npz'), **raw_arrays)
    B.write(output / (family + '.json'), {'records': all_rows, 'geometry_invariance_checks': checks,
            'source_sha256': B.sha(__file__), 'protocol_sha256': B.sha(config),
            'model_frames': len(records), 'original_metric_parity_max_error': original_error,
            'raw_sha256': B.sha(output / (family + '-errors.npz')),
            'max_feature_plane_output_change_under_projection': max(row['max_output_error'] for row in checks)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['orthogonal', 'coupled'], required=True)
    with torch.inference_mode(): main(parser.parse_args().family)
