"""Measure geometry and fixed-target recovery at every frozen toy checkpoint."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
import analyze_toy_cis as FIT
import characterization_core as C

torch.set_num_threads(2)
AN = C.OUT / 'analysis'


def prepare():
    assert not (C.OUT / 'analysis-plan.json').exists()
    inputs = {}
    with np.load(C.OUT / 'assets.npz') as assets:
        for key in ['encoder', 'background_bits', 'background_context', 'random_observation_pair',
                    'recovery_bits', 'recovery_context', 'recovery_noise']:
            inputs[key] = assets[key].copy()
    old = C.ROOT / 'reports/toy-scalar-comparison-v1'
    with np.load(old / 'geometry-inputs.npz') as assets:
        inputs['historical_backgrounds'] = assets['backgrounds'].copy()
    rng = np.random.default_rng(C.P['recovery']['data_seed'])
    n = C.P['recovery']['samples']
    inputs['historical_x'] = (rng.random((n, 100)) < .02) * rng.uniform(-1, 1, (n, 100))
    inputs['historical_noise'] = np.random.default_rng(C.P['recovery']['noise_seed']).normal(size=(2, n, 100))
    rng = np.random.default_rng(C.P['geometry']['selection_seed'])
    random = rng.normal(size=(2, 100)); random[0] /= np.linalg.norm(random[0])
    random[1] -= random[1] @ random[0] * random[0]; random[1] /= np.linalg.norm(random[1])
    inputs['historical_random_pair'] = random
    np.savez_compressed(C.OUT / 'analysis-inputs.npz', **inputs)
    networks = []
    for step in C.P['training']['checkpoints']:
        for name in C.P['activations']:
            for member, (task, seed) in enumerate(C.members()):
                networks.append({'key': f'new_{name}_{task}_s{seed}_t{step:06d}', 'track': 'new',
                                 'activation': name, 'task': task, 'seed': seed, 'step': step,
                                 'member': member, 'weight_group': name,
                                 'weights': f'reports/toy-characterization-v1/weights-{step:06d}.safetensors',
                                 'evaluation': f'reports/toy-characterization-v1/eval-{step:06d}.json'})
    aliases = {'relu': 'relu50', 'gelu': 'gelu50', 'silu': 'silu50', 'leaky_relu': 'leakyrelu50', 'relu2': 'relu2_50'}
    for step in C.P['historical_track']['checkpoints']:
        for name, group in aliases.items():
            cohort = 'toy-scalar-additions-v1' if name in ['leaky_relu', 'relu2'] else 'toy-scalars-core-v1'
            for member, (loss, seed) in enumerate([(loss, seed) for loss in [2, 4] for seed in [0, 1, 2]]):
                networks.append({'key': f'historical_{name}_L{loss}_s{seed}_t{step:06d}', 'track': 'historical',
                                 'activation': name, 'task': f'L{loss}', 'seed': seed, 'step': step,
                                 'member': member, 'weight_group': group,
                                 'weights': f'reports/{cohort}/weights-{step:06d}.safetensors',
                                 'evaluation': f'reports/{cohort}/eval-{step:06d}.json'})
    C.write(C.OUT / 'analysis-plan.json', {'networks': networks, 'source_sha256': C.sha(__file__),
            'protocol_sha256': C.sha(C.ROOT / 'configs/toy-characterization-v1.json'),
            'inputs_sha256': C.sha(C.OUT / 'analysis-inputs.npz'),
            'selection': 'All fixed networks, checkpoints and pairs; no p-based inclusion or promotion.'})
    print('PREPARED', len(networks), 'checkpoint/network records', flush=True)


def model_from(meta):
    weights = load_file(C.ROOT / meta['weights'])
    result = {role: torch.tensor(weights[f'{meta["weight_group"]}.{role}'][meta['member']], dtype=torch.float64)
              for role in ['gate', 'out']}
    if meta['track'] == 'new':
        result.update({role: torch.tensor(weights[f'{meta["weight_group"]}.{role}'][meta['member']], dtype=torch.float64)
                       for role in ['bias', 'out_bias']})
    else:
        result.update(bias=torch.zeros(50, dtype=torch.float64), out_bias=torch.zeros(100, dtype=torch.float64))
    return result


def separable(x, model, meta, encoder, outputs=None):
    # Restrict each output to its own known latent coordinate, preserving its full curve.
    indices = list(range(model['out'].shape[0])) if outputs is None else list(outputs)
    if meta['track'] == 'historical':
        values = x[:, indices]
        projections = model['gate'][:, indices].T
        h = C.activation(values[..., None] * projections[None] + model['bias'][None, None], meta['activation'])
        return (h * model['out'][indices][None]).sum(-1) + model['out_bias'][indices]
    result = torch.empty((len(x), len(indices)), dtype=torch.float64)
    for sign in [-1., 1.]:
        selected = x[:, -1] == sign
        if not selected.any(): continue
        a = torch.tensor(C.effective_encoder(encoder, meta['task'], sign), dtype=torch.float64)
        u = (x[selected, :32] @ a)[:, indices]
        projections = (model['gate'][:, :32] @ a)[:, indices].T
        baseline = model['bias'] + sign * model['gate'][:, -1]
        h = C.activation(u[..., None] * projections[None] + baseline[None, None], meta['activation'])
        result[selected] = (h * model['out'][indices][None]).sum(-1) + model['out_bias'][indices]
    return result


def predict(x, model, meta, encoder, which, outputs=None):
    if which == 'natural': return C.forward(x, model, meta['activation'])
    return separable(x, model, meta, encoder, outputs)


def geometry_case(model, meta, assets, pair, anchor, sign, kind, which, n):
    axis = np.linspace(0, 1.5, n); aa, bb = np.meshgrid(axis, axis, indexing='ij')
    coefficients = torch.tensor(np.stack([sign[0] * aa.ravel(), sign[1] * bb.ravel()], -1), dtype=torch.float64)
    rows = []
    selected_outputs = pair if which == 'separable' and kind == 'feature_axes' else None
    for index in range(4):
        if meta['track'] == 'new':
            z = assets['background_bits'][index].copy(); z[pair] = anchor
            context = float(assets['background_context'][index]); a = C.effective_encoder(assets['encoder'], meta['task'], context)
            base = torch.tensor(np.r_[a @ z, context][None], dtype=torch.float64)
            direction = np.c_[a[:, pair].T if kind == 'feature_axes' else assets['random_observation_pair'], np.zeros(2)]
        else:
            z = assets['historical_backgrounds'][index].copy(); z[pair] = anchor
            base = torch.tensor(z[None], dtype=torch.float64)
            direction = np.eye(100)[pair] if kind == 'feature_axes' else assets['historical_random_pair']
        delta = coefficients @ torch.tensor(direction, dtype=torch.float64)
        clean = predict(base, model, meta, assets['encoder'], which, selected_outputs)
        changed = predict(base + delta, model, meta, assets['encoder'], which, selected_outputs)
        response = (changed - clean).norm(dim=-1).numpy().reshape(n, n)
        assert np.isfinite(response).all() and response[0, 0] < 1e-10
        rows.append(response)
    return np.stack(rows)


def measure_geometry(model, meta, assets):
    pairs = C.P['geometry']['fixed_pairs'] if meta['track'] == 'new' else C.P['historical_track']['pairs']
    records = []; grids = []; fine = []; fine_records = []
    for pi, pair in enumerate(pairs):
        for anchor in C.P['geometry']['pair_anchor_values']:
            for sign in C.P['geometry']['signs']:
                for kind in C.P['geometry']['direction_kinds']:
                    for which in ['natural', 'separable']:
                        response = geometry_case(model, meta, assets, pair, anchor, sign, kind, which, C.P['geometry']['grid_points'])
                        median = np.median(response, axis=0)
                        fits = {str(fraction): FIT.fit(median, fraction)[0] for fraction in C.P['geometry']['threshold_fractions']}
                        row = {'pair': pair, 'anchor': anchor, 'sign': sign, 'kind': kind, 'map': which, 'fits': fits,
                               'per_background': [FIT.fit(grid, .5)[0] for grid in response]}
                        records.append(row); grids.append(response)
                        if meta['track'] == 'new' and meta['step'] == 100000 and meta['seed'] == 0 and pi == 0 and anchor == 0 and sign == [1, 1] and kind == 'feature_axes':
                            finer = geometry_case(model, meta, assets, pair, anchor, sign, kind, which, 61)
                            fit = FIT.fit(np.median(finer, axis=0), .5)[0]
                            delta = fit['p'] - fits['0.5']['p'] if fit['p'] is not None and fits['0.5']['p'] is not None else None
                            fine.append(finer); fine_records.append({'map': which, 'fit': fit, 'coarse_p': fits['0.5']['p'], 'delta_p': delta})
    return records, np.stack(grids), fine_records, np.stack(fine) if fine else np.empty((0, 4, 61, 61))


def recovery_inputs(meta, assets, sigma):
    if meta['track'] == 'new':
        bits = assets['recovery_bits']; context = assets['recovery_context']
        a = torch.tensor(assets['encoder'], dtype=torch.float64)
        base = C.task_inputs(torch.tensor(bits, dtype=torch.float64), torch.tensor(context, dtype=torch.float64), a, meta['task']).numpy()
        corrupted = np.broadcast_to(base, (2, *base.shape)).copy()
        corrupted[..., :32] += sigma * assets['recovery_noise']
        references = {}
        for draw in range(2):
            linear, oracle = C.posterior_mean(corrupted[draw, :, :32], assets['encoder'], context, meta['task'], sigma)
            references.setdefault('linear_reference', []).append(linear); references.setdefault('Bayes_oracle', []).append(oracle)
        references = {key: np.stack(value) for key, value in references.items()}
        return base, corrupted, bits.astype(float), bits > 0, bits == 0, np.zeros_like(bits, dtype=bool), references
    base = assets['historical_x']; corrupted = base[None] + sigma * assets['historical_noise']
    target = np.maximum(base, 0)
    return base, corrupted, target, base > 0, base == 0, base < 0, {'task_reference': np.maximum(corrupted, 0)}


def recovery(model, meta, assets):
    levels = C.P['recovery']['noise_std'] if meta['track'] == 'new' else [0, .075, .15]
    records = []; raw = {}
    for level_index, sigma in enumerate(levels):
        base, corrupted, target, positive, inactive, negative, references = recovery_inputs(meta, assets, sigma)
        flat = torch.tensor(corrupted.reshape(-1, corrupted.shape[-1]), dtype=torch.float64)
        predictions = dict(references)
        clean_predictions = {}
        for which in ['natural', 'separable']:
            pieces = [predict(batch, model, meta, assets['encoder'], which).numpy() for batch in flat.split(256)]
            predictions[which] = np.concatenate(pieces).reshape(2, *target.shape)
            clean_predictions[which] = predict(torch.tensor(base), model, meta, assets['encoder'], which).numpy()
        for which, prediction in predictions.items():
            assert np.isfinite(prediction).all()
            error = (prediction - target) ** 2
            arrays = {'all_sum': error.sum(-1).mean(0), 'all_count': np.full(len(base), target.shape[-1])}
            for label, mask in [('active', positive), ('inactive', inactive), ('negative', negative)]:
                arrays[label + '_sum'] = (error * mask).sum(-1).mean(0); arrays[label + '_count'] = mask.sum(-1)
            miss = prediction < (.5 if meta['track'] == 'new' else .5 * target)
            false_positive = prediction > (.5 if meta['track'] == 'new' else .1)
            arrays['miss_sum'] = (miss * positive).sum(-1).mean(0)
            arrays['false_positive_sum'] = (false_positive * inactive).sum(-1).mean(0)
            row = {'sigma': sigma, 'map': which}
            for label in ['all', 'active', 'inactive', 'negative']:
                count = arrays[label + '_count'].sum()
                row[label + '_mse'] = float(arrays[label + '_sum'].sum() / count) if count else None
            row['active_miss_rate'] = float(arrays['miss_sum'].sum() / arrays['active_count'].sum())
            row['inactive_false_positive_rate'] = float(arrays['false_positive_sum'].sum() / arrays['inactive_count'].sum())
            if which in clean_predictions:
                row['clean_output_mse'] = float(np.square(prediction - clean_predictions[which]).mean())
            records.append(row)
            raw.update({f'{level_index}_{which}__{key}': value for key, value in arrays.items()})
    return records, raw


def measure(track, step=None, limit=None):
    plan = json.loads((C.OUT / 'analysis-plan.json').read_text())
    assert plan['source_sha256'] == C.sha(__file__)
    assert plan['inputs_sha256'] == C.sha(C.OUT / 'analysis-inputs.npz')
    with np.load(C.OUT / 'analysis-inputs.npz') as inputs: assets = {key: inputs[key].copy() for key in inputs.files}
    directory = AN / track; directory.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in plan['networks'] if row['track'] == track and (step is None or row['step'] == step)]
    if limit is not None: candidates = candidates[:limit]
    started = time.perf_counter()
    for meta in candidates:
        target = directory / (meta['key'] + '.json')
        dependencies = {'source_sha256': C.sha(__file__), 'core_sha256': C.sha(C.__file__),
                        'fit_helper_sha256': C.sha(FIT.__file__), 'inputs_sha256': plan['inputs_sha256'],
                        'weights_sha256': C.sha(C.ROOT / meta['weights'])}
        if target.exists():
            assert json.loads(target.read_text())['dependencies'] == dependencies
            continue
        model = model_from(meta)
        evaluation = json.loads((C.ROOT / meta['evaluation']).read_text())['networks']
        row = next(row for row in evaluation if row['activation'] == meta['activation'] and row['seed'] == meta['seed']
                   and (row['task'] == meta['task'] if track == 'new' else row['loss_exponent'] == int(meta['task'][1:])))
        cases, grids, fine, fine_grids = measure_geometry(model, meta, assets)
        scores, error_arrays = recovery(model, meta, assets)
        raw_path = directory / (meta['key'] + '.npz')
        np.savez_compressed(raw_path, response=grids, fine_response=fine_grids, **error_arrays)
        C.write(target, {'network': meta, 'task_evaluation': row, 'geometry': cases, 'fine_checks': fine,
                         'recovery': scores, 'dependencies': dependencies, 'raw_sha256': C.sha(raw_path)})
        print('CHARACTERIZED', meta['key'], 'elapsed', round(time.perf_counter() - started, 1), flush=True)
    marker = f'analysis-complete-{track}' + ('' if step is None else f'-{step:06d}') + '.json'
    C.write(C.OUT / marker,
            {'track': track, 'checkpoint': step, 'candidates': len(candidates), 'seconds': time.perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'measure'], required=True)
    parser.add_argument('--track', choices=['new', 'historical'], default='new')
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    with torch.inference_mode():
        prepare() if args.stage == 'prepare' else measure(args.track, args.checkpoint, args.limit)
