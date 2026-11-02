"""Independent checkpoint, CPU/NumPy parity and raw-metric audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
import torch
import characterization_core as B
from check_characterization import numpy_forward


def main(family, training_only=False):
    if family == 'orthogonal': C = B
    else:
        import coupled_characterization as C
    out = C.OUT
    done = json.loads((out / 'TRAINING-COMPLETE.json').read_text())
    seal = json.loads((out / 'training-input-seal.json').read_text())
    deps = done['dependencies']
    assert done['steps'] == 100000 and done['networks'] == 60
    assert all(seal[key] == value for key, value in deps.items())
    assert deps['core_source_sha256'] == C.sha(C.__file__)
    assert deps['source_sha256'] == C.sha(B.ROOT / 'scripts/train_characterization.py')
    assert deps['protocol_sha256'] == C.sha(B.ROOT / 'configs' / (C.P['version'] + '.json'))
    assert deps['assets_sha256'] == C.sha(out / 'assets.npz')
    logs = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert [row['step'] for row in logs] == [1, 100] + list(range(1000, 100001, 1000))
    assert logs[-1]['lr'] == 0
    assert all(np.isfinite(values).all() for row in logs for values in row['losses'].values())
    initial = load_file(out / 'weights-000000.safetensors')
    final = load_file(out / 'weights-100000.safetensors')
    groups, _ = C.initialize()
    for name, group in groups.items():
        for role, expected in group.items():
            assert np.array_equal(initial[name + '.' + role], expected.detach().numpy())
            changes = np.linalg.norm((final[name + '.' + role] - initial[name + '.' + role]).reshape(12, -1), axis=1)
            assert (changes > 1e-6).all()
    assets = dict(np.load(out / 'assets.npz'))
    parity_errors = []
    for step in C.P['training']['checkpoints']:
        weights = load_file(out / f'weights-{step:06d}.safetensors')
        evaluation = json.loads((out / f'eval-{step:06d}.json').read_text())
        assert evaluation['dependencies'] == deps and len(evaluation['networks']) == 60
        with np.load(out / f'prediction-check-{step:06d}.npz') as cached:
            for name in C.P['activations']:
                for index, (task, seed) in enumerate(C.members()):
                    z = assets['eval_bits'][:128]; context = assets['eval_context'][:128]
                    a = assets['encoder'].astype(np.float32)
                    obs = z @ a.T
                    if task.startswith('context'):
                        alternate = z @ a[:, np.roll(np.arange(16), 8)].T
                        obs = np.where(context[:, None] < 0, alternate, obs)
                    x = np.c_[obs, context]
                    model = {role: torch.tensor(weights[name + '.' + role][index]) for role in ['gate', 'out', 'bias', 'out_bias']}
                    expected = numpy_forward(x, model, name)
                    error = float(np.abs(expected - cached[name][index]).max())
                    assert error < 5e-5, (family, step, name, task, seed, error)
                    parity_errors.append(error)
    results = {'all_assertions_passed': True, 'family': family, 'trained_networks': 60,
               'updates_each': 100000, 'checkpoint_frames': 360, 'log_records': len(logs),
               'independent_numpy_vs_saved_MPS_prediction_max_error': max(parity_errors),
               'all_role_parameters_changed_in_all_networks': True, 'source_sha256': C.sha(__file__)}
    if family == 'coupled':
        parent = json.loads((B.OUT / 'TRAINING-COMPLETE.json').read_text())
        assert parent['final_rng_sha256'] == done['final_rng_sha256']
        parent_initial = load_file(B.OUT / 'weights-000000.safetensors')
        assert all(np.array_equal(initial[key], parent_initial[key]) for key in initial)
        results['exact_parent_initialization_and_final_minibatch_RNG_pairing'] = True
    if not training_only:
        plan = json.loads((out / 'analysis-plan.json').read_text())
        records = list((out / 'analysis').glob('*/*.json'))
        assert len(records) == len(plan['networks'])
        plan_by_key = {row['key']: row for row in plan['networks']}
        primary_count = flagged_count = invalid_count = 0; fine = []; points = 0
        for path in records:
            record = json.loads(path.read_text()); meta = record['network']
            assert meta == plan_by_key[path.stem]
            assert C.sha(B.ROOT / meta['weights']) == record['dependencies']['weights_sha256']
            assert C.sha(path.with_suffix('.npz')) == record['raw_sha256']
            assert record['dependencies']['inputs_sha256'] == C.sha(out / 'analysis-inputs.npz')
            assert record['dependencies']['core_sha256'] == C.sha(C.__file__)
            with np.load(path.with_suffix('.npz')) as raw:
                for key in ['response', 'fine_response']:
                    value = raw[key]; assert np.isfinite(value).all() and (value >= 0).all()
                    if value.size: assert value[..., 0, 0].max() < 1e-9
                    points += value.size
                assert raw['response'].shape == (32, 4, 31, 31)
                for row in record['recovery']:
                    levels = C.P['recovery']['noise_std'] if meta['track'] == 'new' else [0, .075, .15]
                    li = levels.index(row['sigma']); prefix = f'{li}_{row["map"]}__'
                    for key in ['all', 'active', 'inactive', 'negative']:
                        sums = raw[prefix + key + '_sum']; counts = raw[prefix + key + '_count']
                        assert np.isfinite(sums).all() and (sums >= 0).all()
                        pooled = float(sums.sum() / counts.sum()) if counts.sum() else None
                        assert pooled == row[key + '_mse']
                    total = sum(raw[prefix + key + '_count'] for key in ['active', 'inactive', 'negative'])
                    assert np.array_equal(total, raw[prefix + 'all_count'])
                    subtotal = sum(raw[prefix + key + '_sum'] for key in ['active', 'inactive', 'negative'])
                    assert np.max(np.abs(subtotal - raw[prefix + 'all_sum'])) < 1e-8
            for row in record['geometry']:
                fit = row['fits']['0.5']; primary_count += 1
                flagged_count += bool(fit['flags']); invalid_count += fit['p'] is None
            fine.extend(row for row in record['fine_checks'])
        results.update(analysis_records=len(records), expected_analysis_records=len(plan['networks']),
                       aggregate_surfaces=primary_count, primary_fits_flagged=flagged_count,
                       primary_fits_invalid=invalid_count, response_points=points,
                       fine_check_count=len(fine), fine_max_abs_delta_p=max(abs(row['delta_p']) for row in fine if row['delta_p'] is not None),
                       all_raw_sums_counts_and_pooled_metrics_consistent=True)
    B.write(out / ('training-audit.json' if training_only else 'quality-audit.json'), results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['orthogonal', 'coupled'], default='orthogonal')
    parser.add_argument('--training-only', action='store_true')
    args = parser.parse_args(); main(args.family, args.training_only)
