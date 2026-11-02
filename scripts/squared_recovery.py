"""Tiny squared-ReLU independent-control and fixed-target corruption assay.

CPU float64; no training. Freeze --stage prepare before --stage measure.
See configs/squared-recovery-v1.json for the scope and prespecified settings.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/squared-recovery-v1.json'
P = json.loads(PROTOCOL.read_text())
SOURCE = ROOT / 'reports' / P['source_report']
torch.set_num_threads(2)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def forward(x, model):
    return (x @ model['gate'].T).relu().square() @ model['out'].T


def prepare(out):
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}; choose a new --out.')
    out.mkdir(parents=True)
    evaluation_path = SOURCE / 'eval-100000.json'
    evaluation = next(row for row in json.loads(evaluation_path.read_text())['networks']
                      if row['id'] == P['trained_id'])
    plan_path = SOURCE / 'geometry-plan.json'
    network = next(row for row in json.loads(plan_path.read_text())['networks']
                   if row['id'] == P['trained_id'])
    weights_path = SOURCE / 'weights-100000.safetensors'
    arrays = load_file(weights_path)
    exponent_seed = [(2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2)]
    member = exponent_seed.index((evaluation['loss_exponent'], evaluation['seed']))
    trained = {role: tensor(arrays[f'relu2_50.{role}'][member]) for role in ('gate', 'out')}
    eligible = np.array(sorted(evaluation['competent_feature_indices']), dtype=np.int64)
    assert len(eligible) == trained['gate'].shape[0] == 50
    single = forward(torch.eye(100, dtype=torch.float64), trained)
    gains = single.diagonal()[eligible]
    norms = trained['gate'][:, eligible].norm(dim=0)
    assert (gains > 0).all() and (norms > 0).all()
    independent = {role: torch.zeros_like(value) for role, value in trained.items()}
    for unit, feature in enumerate(eligible):
        independent['gate'][unit, feature] = norms[unit]
        independent['out'][feature, unit] = gains[unit] / norms[unit].square()
    control_single = forward(torch.eye(100, dtype=torch.float64), independent)
    gain_error = float((control_single.diagonal()[eligible] - gains).abs().max())
    norm_error = float((independent['gate'][:, eligible].norm(dim=0) - norms).abs().max())
    assert gain_error < 1e-10 and norm_error < 1e-10
    for amplitude in [.25, .5, 1.]:
        assert float((forward(amplitude * torch.eye(100, dtype=torch.float64), trained).diagonal()[eligible]
                      - forward(amplitude * torch.eye(100, dtype=torch.float64), independent).diagonal()[eligible]).abs().max()) < 1e-10
    weights = {f'{name}_{role}': value.numpy() for name, model in
               [('trained', trained), ('independent', independent)] for role, value in model.items()}
    np.savez_compressed(out / 'models.npz', **weights)
    c = P['corruption']
    rng = np.random.default_rng(c['data_seed'])
    clean = (rng.random((c['samples'], 100)) < c['activation_probability']) * rng.uniform(-1, 1, (c['samples'], 100))
    noise = np.random.default_rng(c['noise_seed']).normal(size=(c['noise_draws_per_sample'], c['samples'], 100))
    with np.load(SOURCE / 'geometry-inputs.npz') as original:
        backgrounds = original['backgrounds'].copy()
    np.savez_compressed(out / 'inputs.npz', clean_x=clean, standard_noise=noise,
                        supported=eligible, backgrounds=backgrounds,
                        pair=np.asarray(network['conditioned_pairs'][0]))
    proof = {
        'trained_id': evaluation['id'], 'pair': network['conditioned_pairs'][0],
        'supported_features': eligible.tolist(), 'width': 50, 'input_output_features': 100,
        'trained_mean_competent_features_per_neuron': evaluation['mean_neuron_feature_participation'],
        'control_nonzero_encoder_weights': int(torch.count_nonzero(independent['gate'])),
        'control_nonzero_decoder_weights': int(torch.count_nonzero(independent['out'])),
        'max_positive_single_feature_gain_error': gain_error,
        'max_supported_encoder_column_norm_error': norm_error,
        'model_weights_sha256': sha(weights_path), 'evaluation_sha256': sha(evaluation_path),
        'geometry_plan_sha256': sha(plan_path), 'source_sha256': sha(__file__),
        'protocol_sha256': sha(PROTOCOL), 'models_sha256': sha(out / 'models.npz'),
        'inputs_sha256': sha(out / 'inputs.npz'),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'selection': 'One already-known specialized checkpoint; first saved competent pair. No new p or corruption outcomes used.',
        'match_limits': 'Same width, supported features, positive isolated-feature self-response and encoder-column norms. Full clean outputs/cross-talk, negative rays, pair cosine and number of free connections are not exactly matched.'
    }
    write(out / 'prepared.json', proof)
    print('PREPARED', proof['trained_id'], 'pair', proof['pair'], 'matched features', len(eligible), flush=True)


def directions(model, pair):
    columns = model['gate'][:, pair].T.numpy()
    norms = np.linalg.norm(columns, axis=1)
    rho = float(columns[0] @ columns[1] / np.prod(norms))
    assert abs(rho) < .999999
    rng = np.random.default_rng(P['geometry']['random_seed'])
    u = rng.normal(size=50); u /= np.linalg.norm(u)
    v = rng.normal(size=50); v -= v @ u * u; v /= np.linalg.norm(v)
    random = np.stack([norms[0] * u, norms[1] * (rho * u + np.sqrt(1 - rho * rho) * v)])
    error = float(np.abs(columns @ columns.T - random @ random.T).max())
    assert error < 1e-9
    return {'encoder_columns': columns, 'gram_matched_random': random}, {'cosine': rho, 'gram_error': error}


def geometry(models, inputs, out):
    import analyze_toy_cis as fitter
    config = P['geometry']; pair = inputs['pair'].tolist(); n = config['grid_points']
    axis = np.linspace(0, config['amplitude_max'], n)
    a, b = np.meshgrid(axis, axis, indexing='ij')
    coefficients = tensor(np.stack([a.ravel(), b.ravel()], axis=1))
    rows = []; grids = []; checks = {}
    for name, model in models.items():
        ds, checks[name] = directions(model, pair)
        for kind, direction in ds.items():
            delta = coefficients @ tensor(direction)
            for anchor in config['anchor_values']:
                x = tensor(inputs['backgrounds'].copy()); x[:, pair] = anchor
                z = x @ model['gate'].T; clean = z.relu().square() @ model['out'].T
                response = np.stack([((base[None] + delta).relu().square() @ model['out'].T - ref).norm(dim=-1).numpy().reshape(n, n)
                                     for base, ref in zip(z, clean)])
                assert np.isfinite(response).all() and response[:, 0, 0].max() < 1e-10
                median = np.median(response, axis=0)
                fits = {str(f): fitter.fit(median, f)[0] for f in config['threshold_fractions']}
                per = [fitter.fit(grid, config['primary_fraction'])[0] for grid in response]
                rows.append({'model': name, 'kind': kind, 'anchor_value': anchor,
                             'features': pair, 'sign': [1, 1], 'fits': fits, 'per_background': per})
                grids.append(response)
    np.savez_compressed(out / 'geometry.npz', axis=axis, l2=np.stack(grids))
    write(out / 'geometry.json', {'records': rows, 'direction_checks': checks, 'raw_sha256': sha(out / 'geometry.npz')})
    print('GEOMETRY', len(rows), 'surfaces', flush=True)


def error_arrays(prediction, clean_prediction, target, x, supported):
    """One row per clean example, averaged over its paired noise repetitions."""
    error = (prediction - target).square()
    supported_error = error[..., supported]
    s = x[:, supported]
    result = {'all_sum': supported_error.sum(-1).mean(0).numpy(),
              'all_count': np.full(len(x), len(supported)),
              'full_sum': error.sum(-1).mean(0).numpy(), 'full_count': np.full(len(x), 100),
              'clean_output_sum': (prediction[..., supported] - clean_prediction[:, supported]).square().sum(-1).mean(0).numpy()}
    for name, mask in [('active', s > 0), ('absent', s == 0), ('negative', s < 0)]:
        result[name + '_sum'] = (supported_error * mask).sum(-1).mean(0).numpy()
        result[name + '_count'] = mask.sum(-1).numpy()
    result['false_positive_sum'] = ((prediction[..., supported] > .1) * (s == 0)).sum(-1).double().mean(0).numpy()
    return result


def metrics(arrays):
    result = {name + '_mse': float(arrays[name + '_sum'].sum() / arrays[name + '_count'].sum())
              for name in ('all', 'full', 'active', 'absent', 'negative')}
    result['absent_fraction_above_0p1'] = float(arrays['false_positive_sum'].sum() / arrays['absent_count'].sum())
    result['clean_output_mse'] = float(arrays['clean_output_sum'].sum() / arrays['all_count'].sum())
    return result


def corruption(models, inputs, out):
    c = P['corruption']; x = tensor(inputs['clean_x']); target = x.relu()
    noise = tensor(inputs['standard_noise']); supported = inputs['supported'].tolist()
    raw = {}; records = []; dictionaries = {}
    for level_index, sigma in enumerate(c['coordinate_noise_std']):
        delta = sigma * noise; corrupted = x[None] + delta
        predictions = [('input_reference', 'task_on_corrupted_input', corrupted.relu(), target)]
        for name, model in models.items():
            clean = forward(x, model); z = x @ model['gate'].T
            latent_delta = delta @ model['gate'].T
            natural = (z[None] + latent_delta).relu().square() @ model['out'].T
            oracle = clean[None] + (2 * z.relu()[None] * latent_delta) @ model['out'].T
            direct = forward(corrupted, model)
            assert float((natural - direct).abs().max()) < 1e-10
            # Exact first derivative and a zero-noise check, without a noise outcome gate.
            if sigma == 0:
                assert float((oracle - natural).abs().max()) < 1e-10
            predictions += [(name, 'natural_squared_relu', natural, clean),
                            (name, 'clean_anchor_jacobian_oracle', oracle, clean)]
        for name, method, prediction, clean in predictions:
            arrays = error_arrays(prediction, clean, target, x, supported)
            prefix = f'{level_index}_{name}_{method}'
            raw.update({prefix + '__' + key: value for key, value in arrays.items()})
            dictionaries[(sigma, name, method)] = arrays
            row = {'model': name, 'method': method, 'sigma': sigma, **metrics(arrays)}
            row['fraction_corrupted_input_abs_over_1'] = float((corrupted.abs() > 1).double().mean())
            records.append(row)
    paired = []
    rng = np.random.default_rng(c['bootstrap']['seed'])
    resamples = rng.integers(0, len(x), size=(c['bootstrap']['replicates'], len(x)))
    for sigma in c['coordinate_noise_std']:
        for metric in ('all', 'active', 'absent', 'negative'):
            trained = dictionaries[(sigma, 'trained', 'natural_squared_relu')]
            for other_name, other_method, label in (
                    ('independent', 'natural_squared_relu', 'trained_minus_independent'),
                    ('input_reference', 'task_on_corrupted_input', 'trained_minus_input_reference'),
                    ('trained', 'clean_anchor_jacobian_oracle', 'trained_minus_clean_anchor_oracle')):
                other = dictionaries[(sigma, other_name, other_method)]
                difference = trained[metric + '_sum'] - other[metric + '_sum']
                count = trained[metric + '_count']
                assert np.array_equal(count, other[metric + '_count'])
                estimates = difference[resamples].sum(1) / count[resamples].sum(1)
                paired.append({'sigma': sigma, 'metric': metric, 'comparison': label,
                               'mse_difference': float(difference.sum() / count.sum()),
                               'bootstrap_95pct': np.quantile(estimates, [.025, .975]).tolist()})
    np.savez_compressed(out / 'corruption-errors.npz', **raw)
    write(out / 'corruption.json', {'records': records, 'paired_differences': paired,
                                   'samples': len(x), 'draws_per_sample': noise.shape[0],
                                   'raw_sha256': sha(out / 'corruption-errors.npz'),
                                   'target': c['target'],
                                   'uncertainty': 'Example-block bootstrap conditional on one trained checkpoint, one input seed and this noise set; not training-seed uncertainty.'})
    print('CORRUPTION', len(records), 'model/method/noise-level records', flush=True)


def measure(out):
    if (out / 'COMPLETE.json').exists() or (out / 'geometry.npz').exists() or (out / 'corruption.json').exists():
        raise ValueError('Measured outputs already exist; use a new prepared run.')
    proof = json.loads((out / 'prepared.json').read_text())
    for path, key in [(Path(__file__), 'source_sha256'), (PROTOCOL, 'protocol_sha256'),
                      (out / 'models.npz', 'models_sha256'), (out / 'inputs.npz', 'inputs_sha256')]:
        assert sha(path) == proof[key], f'Prepared input/source changed: {path}'
    with np.load(out / 'models.npz') as weights:
        models = {name: {role: tensor(weights[f'{name}_{role}'].copy()) for role in ('gate', 'out')}
                  for name in ('trained', 'independent')}
    with np.load(out / 'inputs.npz') as arrays:
        inputs = {key: arrays[key].copy() for key in arrays.files}
    started = time.perf_counter()
    geometry(models, inputs, out)
    corruption(models, inputs, out)
    write(out / 'COMPLETE.json', {'all_runtime_assertions_passed': True, 'source_sha256': sha(__file__),
                                'prepared_sha256': sha(out / 'prepared.json'),
                                'seconds': time.perf_counter() - started, 'torch': torch.__version__,
                                'numpy': np.__version__, 'device': 'CPU float64',
                                'scope': 'One trained specialized checkpoint and one analytic no-cross-feature control; no new training.'})
    print('MEASUREMENT_COMPLETE', round(time.perf_counter() - started, 2), 'seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'measure'], required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/squared-recovery-v1')
    args = parser.parse_args()
    with torch.inference_mode():
        globals()[args.stage](args.out.resolve())


if __name__ == '__main__':
    main()
