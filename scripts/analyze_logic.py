"""Small fixed AND/XOR geometry and recovery sweep; preserve every failed fit."""
import argparse
import json
import time

import numpy as np
from scipy.ndimage import label
from safetensors.numpy import load_file
import analyze_toy_cis as FIT
import logic_core as C


def fit(grid, fraction=.5):
    result, _ = FIT.fit(grid, fraction)
    if 'intercepts_feature_units' in result:
        result['intercepts_feature_units'] = np.asarray(result['intercepts_feature_units']) * C.P['geometry']['amplitude_max'] / 1.5
    threshold = result['threshold']; maximum = float(grid.max())
    extra = []
    labels, count = label(grid < threshold)
    sizes = np.bincount(labels.ravel())[1:]
    components = int((sizes >= 3).sum())
    diagonal = np.diag(grid)
    # A tangent touch counts: exact XOR reaches the half-axis threshold at the
    # middle of its diagonal, then returns to the original output at (1,1).
    above = np.where(diagonal >= threshold * (1 - 1e-6))[0]
    recross = bool(len(above) and (diagonal[above[0] + 1:] < threshold * (1 - 1e-6)).any())
    drop = max(float(np.maximum(-np.diff(grid, axis=axis), 0).max()) for axis in [0, 1]) / max(maximum, 1e-15)
    if threshold > 1e-10:
        if components > 1: extra.append('multiple_below_threshold_components')
        if recross: extra.append('diagonal_recross_below_threshold')
        if drop > .01: extra.append('interior_nonmonotone_over_1pct')
    result.update(assay_flags=extra, all_flags=list(result['flags']) + extra,
                  below_threshold_components=components, diagonal_recross=recross,
                  max_interior_step_drop_fraction=drop,
                  min_axis_to_full_grid_range=min(result['axis_max_response']) / max(maximum, 1e-15))
    return C.BASE.clean(result)


def model_from(meta):
    raw = load_file(C.OUT / f'weights-{meta["step"]:06d}.safetensors')
    return {role: raw[f'{meta["activation"]}.{role}'][meta['member']].astype(float)
            for role in ['gate', 'out', 'bias', 'out_bias']}


def geometry_case(predict, assets, pair, state, n):
    axis = np.linspace(0, C.P['geometry']['amplitude_max'], n)
    aa, bb = np.meshgrid(axis, axis, indexing='ij')
    coefficients = np.stack([aa, bb], -1) * (1 - 2 * np.asarray(state))
    directions = assets['encoder'][:, pair['bits']].T
    responses = []; axes = []; interactions = []; within_outputs = []
    for background in assets['background_bits']:
        z = background.copy(); z[pair['bits']] = state
        base = z @ assets['encoder'].T
        changed = predict(base + coefficients.reshape(-1, 2) @ directions).reshape(n, n, 8)
        delta = changed - predict(base[None])[0]
        norm = np.linalg.norm(delta, axis=-1)
        axis_vectors = np.stack([delta[:, 0], delta[0, :]])
        null = axis_vectors[0][:, None] + axis_vectors[1][None, :]
        responses.append(norm); axes.append(axis_vectors)
        interactions.append(np.linalg.norm(delta - null, axis=-1).astype(np.float32))
        within_outputs.append(np.abs(delta[..., 0]))
    return np.stack(responses), np.stack(axes), np.stack(interactions), np.stack(within_outputs)


def describe_case(response, axes, interactions, within_outputs, pair, state):
    med = np.median(response, axis=0)
    row = {'pair': pair, 'state': state,
           'median_fits': {str(f): fit(med, f) for f in C.P['geometry']['threshold_fractions']},
           'backgrounds': []}
    for i, grid in enumerate(response):
        scale = max(float(grid.max()), 1e-15)
        data = {'background': i, 'fit': fit(grid),
                'mean_vector_interaction_residual': float(interactions[i].mean()) / scale}
        if state == [0, 0] and i == 3:
            null = np.linalg.norm(axes[i, 0][:, None] + axes[i, 1][None, :], axis=-1)
            data['axis_vector_null_fit'] = fit(null)
            if pair['name'] == 'within': data['output0_only_fit'] = fit(within_outputs[i])
        row['backgrounds'].append(data)
    return row


def geometry(predict, assets, fine=False):
    records = []; arrays = {'response': [], 'axis_vectors': [], 'interaction_norm': [], 'output0_response': []}
    n = C.P['geometry']['fine_grid_points' if fine else 'grid_points']
    for pair in C.P['geometry']['pairs']:
        for state in ([[0, 0]] if fine else C.P['geometry']['states']):
            values = geometry_case(predict, assets, pair, state, n)
            records.append(describe_case(*values, pair, state))
            for key, value in zip(arrays, values): arrays[key].append(value)
    return records, {key: np.stack(values) for key, values in arrays.items()}


def recovery(predict, assets, task):
    bits = assets['recovery_bits'].astype(float); clean = bits @ assets['encoder'].T
    target = C.targets(bits, task); records = []; raw = []
    for sigma in C.P['recovery']['noise_std']:
        x = clean[None] + sigma * assets['recovery_noise']
        flat = x.reshape(-1, 32)
        # Quantize once for compact raw storage, then compute metrics from those exact values.
        pred = predict(flat).reshape(2, len(bits), 8).astype(np.float32)
        predictions = {'natural': pred.astype(float), 'clipped_natural': np.clip(pred.astype(float), 0, 1)}
        predictions.update({key: value.reshape(2, len(bits), 8) for key, value in
                            C.references(flat, assets['encoder'], task, sigma).items()})
        for name, prediction in predictions.items():
            records.append({'map': name, 'sigma': sigma, **C.score(prediction, target)})
        raw.append(pred)
    return records, np.stack(raw)


def baselines(assets):
    folder = C.OUT / 'analysis'; folder.mkdir(exist_ok=True)
    path = folder / 'baselines.json'
    if path.exists():
        assert json.loads(path.read_text())['dependencies'] == C.dependencies()
        return
    records = []; arrays = {}
    for task in ['and', 'xor']:
        for which in ['Bayes_oracle', 'plugin_reference']:
            predict = lambda x, task=task, which=which: C.references(x, assets['encoder'], task, .35)[which]
            cases, raw = geometry(predict, assets)
            records.append({'task': task, 'map': which, 'geometry': cases})
            arrays.update({task + '_' + which + '_' + key: value for key, value in raw.items()})
    raw_path = folder / 'baselines.npz'; np.savez_compressed(raw_path, **arrays)
    C.write(path, {'dependencies': C.dependencies(), 'records': records, 'raw_sha256': C.sha(raw_path)})


def main():
    C.assert_seal(); assets = C.load_assets(); folder = C.OUT / 'analysis'; folder.mkdir(exist_ok=True)
    baselines(assets); started = time.perf_counter(); count = 0
    for step in C.P['training']['checkpoints']:
        assert (C.OUT / f'weights-{step:06d}.safetensors').exists(), 'Training must complete before analysis.'
        for meta in C.network_rows(step):
            path = folder / (meta['key'] + '.json')
            dep = {**C.dependencies(), 'weights_sha256': C.sha(C.OUT / f'weights-{step:06d}.safetensors')}
            if path.exists():
                assert json.loads(path.read_text())['dependencies'] == dep
                count += 1; continue
            model = model_from(meta)
            predict = lambda x: C.numpy_forward(x, model, meta['activation'])
            cases, raw = geometry(predict, assets)
            scores, predictions = recovery(predict, assets, meta['task'])
            raw['recovery_predictions'] = predictions
            fine = []
            if step == C.P['training']['steps'] and meta['seed'] == 0:
                fine, fine_raw = geometry(predict, assets, fine=True)
                raw.update({'fine_' + key: value for key, value in fine_raw.items()})
            raw_path = path.with_suffix('.npz'); np.savez_compressed(raw_path, **raw)
            C.write(path, {'network': meta, 'geometry': cases, 'fine_geometry': fine, 'recovery': scores,
                          'dependencies': dep, 'raw_sha256': C.sha(raw_path)})
            count += 1
            print('MEASURED', count, '/180', meta['key'], 'seconds', round(time.perf_counter() - started, 1), flush=True)
    C.write(C.OUT / 'ANALYSIS-COMPLETE.json', {'frames': count, 'dependencies': C.dependencies(),
                                             'seconds': time.perf_counter() - started})


if __name__ == '__main__': main()
