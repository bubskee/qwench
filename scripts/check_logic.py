"""Independent preflight and saved-data audit for the small logical-task pilot."""
import argparse
import io
import json

import numpy as np
from scipy.special import logsumexp
import torch
from safetensors.numpy import load_file
import logic_core as C


def preflight():
    C.assert_seal(); torch.set_num_threads(2)
    import analyze_logic as A
    assets = C.load_assets(); encoder = assets['encoder']
    rng = np.random.default_rng(56201)
    corners = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    bits = np.tile(corners, (1, 8))
    assert np.array_equal(C.targets(bits, 'and')[:, 0], [0, 0, 0, 1])
    assert np.array_equal(C.targets(bits, 'xor')[:, 0], [0, 1, 1, 0])
    assert np.array_equal(C.targets(torch.tensor(bits), 'and').numpy(), C.targets(bits, 'and'))
    assert np.array_equal(C.targets(torch.tensor(bits), 'xor').numpy(), C.targets(bits, 'xor'))
    assert np.abs(encoder.T @ encoder - np.eye(16)).max() < 1e-12
    assert np.abs(assets['train_bits'] @ encoder.T @ encoder - assets['train_bits']).max() < 1e-12
    groups, _ = C.initialize(dtype=torch.float64); checks = {}
    for name, group in groups.items():
        for role, value in group.items():
            assert torch.equal(value, groups['relu'][role])
            for i in range(4): assert torch.equal(value[:3], value[i * 3:(i + 1) * 3])
        x = torch.tensor(rng.normal(size=(12, 7, 32)), dtype=torch.float64)
        y = torch.tensor(rng.normal(size=(12, 7, 8)), dtype=torch.float64)
        batch = {role: value.detach().clone().requires_grad_() for role, value in group.items()}
        optimizer = torch.optim.Adam(list(batch.values()), lr=.001, foreach=True)
        (C.forward(x, batch, name) - y).square().mean((1, 2)).sum().backward()
        gradients = {role: value.grad.clone() for role, value in batch.items()}; optimizer.step()
        grad_error = 0.; update_error = 0.; forward_error = 0.
        for member in range(12):
            single = {role: value[member].detach().clone().requires_grad_() for role, value in group.items()}
            one_opt = torch.optim.Adam(list(single.values()), lr=.001, foreach=True)
            (C.forward(x[member], single, name) - y[member]).square().mean().backward()
            grad_error = max(grad_error, max(float((v.grad - gradients[r][member]).abs().max()) for r, v in single.items()))
            one_opt.step()
            update_error = max(update_error, max(float((v.detach() - batch[r][member].detach()).abs().max()) for r, v in single.items()))
            model = {role: value.detach().numpy() for role, value in single.items()}
            forward_error = max(forward_error, float(np.abs(C.numpy_forward(x[member].numpy(), model, name)
                                - C.forward(x[member], single, name).detach().numpy()).max()))
        assert max(grad_error, update_error, forward_error) < 1e-10
        checks[name] = {'gradient_max_error': grad_error, 'Adam_update_max_error': update_error,
                        'numpy_forward_max_error': forward_error}
    states = np.indices((2,) * 16).reshape(16, -1).T.astype(float)
    observations = states @ encoder.T
    posterior_error = 0.
    for index in range(2):
        observation = encoder @ assets['recovery_bits'][index] + .35 * rng.normal(size=32)
        log_weights = -np.square(observations - observation).sum(1) / (2 * .35 ** 2)
        probabilities = np.exp(log_weights - logsumexp(log_weights))
        for task in ['and', 'xor']:
            enumeration = probabilities @ C.targets(states, task)
            analytic = C.references(observation[None], encoder, task, .35)['Bayes_oracle'][0]
            posterior_error = max(posterior_error, float(np.abs(enumeration - analytic).max()))
    assert posterior_error < 1e-10
    axis = np.linspace(0, 1, 31); a, b = np.meshgrid(axis, axis, indexing='ij')
    calibrations = {}
    for name, grid, expected, intercept in [('linear', np.hypot(a, b), 2., .5),
                                          ('square', np.hypot(a * a, b * b), 4., np.sqrt(.5))]:
        result = A.fit(grid); assert abs(result['p'] - expected) < .04
        assert max(abs(np.array(result['intercepts_feature_units']) - intercept)) < .001
        calibrations[name] = result
    ideal_and = A.fit(a * b); assert 'flat_axis' in ideal_and['flags'] and ideal_and['p'] is None
    ideal_xor = A.fit(np.abs(a + b - 2 * a * b))
    assert ideal_xor['diagonal_recross'] and ideal_xor['below_threshold_components'] == 2
    assert torch.backends.mps.is_available()
    mps_groups, params = C.initialize('mps')
    cpu_x = torch.tensor(rng.normal(size=(12, 8, 32)), dtype=torch.float32)
    mps_error = 0.
    for name, group in mps_groups.items():
        cpu = {role: value.detach().cpu() for role, value in group.items()}
        mps_error = max(mps_error, float((C.forward(cpu_x.to('mps'), group, name).cpu()
                         - C.forward(cpu_x, cpu, name)).abs().max().detach()))
    assert mps_error < 2e-5
    opt = torch.optim.Adam(params, lr=.001, foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, C.P['training']['steps'])
    def step(groups, optimizer, schedule):
        x = torch.randn(12, 8, 32, device='mps'); y = torch.randn(12, 8, 8, device='mps')
        optimizer.zero_grad(set_to_none=True)
        torch.stack([(C.forward(x, group, name) - y).square().mean((1, 2)) for name, group in groups.items()]).sum().backward()
        optimizer.step(); schedule.step()
        return x.cpu(), y.cpu()
    torch.mps.manual_seed(56202)
    for _ in range(3): step(mps_groups, opt, scheduler)
    stream = io.BytesIO()
    torch.save({'weights': {name: {role: v.detach().cpu() for role, v in group.items()} for name, group in mps_groups.items()},
                'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': torch.mps.get_rng_state()}, stream)
    x, y = step(mps_groups, opt, scheduler)
    expected = {name: {role: v.detach().cpu().clone() for role, v in group.items()} for name, group in mps_groups.items()}
    stream.seek(0); saved = torch.load(stream, map_location='cpu', weights_only=True)
    restored = {name: {role: v.to('mps').requires_grad_() for role, v in group.items()} for name, group in saved['weights'].items()}
    opt2 = torch.optim.Adam([v for group in restored.values() for v in group.values()], lr=.001, foreach=True)
    opt2.load_state_dict(saved['optimizer'])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, C.P['training']['steps'])
    sched2.load_state_dict(saved['scheduler']); torch.mps.set_rng_state(saved['rng'])
    xx, yy = step(restored, opt2, sched2); assert torch.equal(x, xx) and torch.equal(y, yy)
    resume_error = max(float((v.detach().cpu() - expected[name][role]).abs().max())
                       for name, group in restored.items() for role, v in group.items())
    assert resume_error == 0
    result = {'all_assertions_passed': True, 'boolean_truth_tables_passed': True,
              'exact_initialization_pairing': True, 'gradient_and_update_checks': checks,
              'posterior_vs_65536_state_enumeration_max_error': posterior_error,
              'MPS_CPU_forward_max_error': mps_error, 'MPS_checkpoint_resume_max_error': resume_error,
              'fit_calibrations': calibrations, 'ideal_AND00_fit': ideal_and, 'ideal_XOR00_fit': ideal_xor,
              'dependencies': C.dependencies()}
    C.write(C.OUT / 'preflight-checks.json', result)
    print('PREFLIGHT_PASS', 'posterior error', posterior_error, 'MPS error', mps_error, 'resume error', resume_error, flush=True)


def audit():
    C.assert_seal(); import analyze_logic as A
    assets = C.load_assets(); frames = 0; parity_error = 0.; score_error = 0.; grid_error = 0.; fit_error = 0.
    fine = []; primary = []; flagged = 0; attempted = 0
    complete = json.loads((C.OUT / 'TRAINING-COMPLETE.json').read_text())
    assert complete['dependencies'] == C.dependencies() and complete['steps'] == 30000 and complete['networks'] == 60
    logs = [json.loads(line) for line in (C.OUT / 'training.jsonl').read_text().splitlines()]
    assert [row['step'] for row in logs] == [1] + list(range(1000, 30001, 1000))
    assert all(np.isfinite(values).all() for row in logs for values in row['losses'].values())
    preflight = json.loads((C.OUT / 'preflight-checks.json').read_text())
    assert preflight['all_assertions_passed'] and preflight['dependencies'] == C.dependencies()
    baselines = json.loads((C.OUT / 'analysis/baselines.json').read_text())
    assert baselines['dependencies'] == C.dependencies()
    assert baselines['raw_sha256'] == C.sha(C.OUT / 'analysis/baselines.npz')
    with np.load(C.OUT / 'analysis/baselines.npz') as raw:
        for record in baselines['records']:
            key = record['task'] + '_' + record['map'] + '_response'
            for index, case in enumerate(record['geometry']):
                for b in range(4):
                    refit = A.fit(raw[key][index, b]); old = case['backgrounds'][b]['fit']
                    assert refit['all_flags'] == old['all_flags'] and refit['p'] == old['p']
    for step in C.P['training']['checkpoints']:
        weights = load_file(C.OUT / f'weights-{step:06d}.safetensors')
        if step == 0:
            initial, _ = C.initialize()
            for name, group in initial.items():
                for role, value in group.items(): assert np.array_equal(value.detach().numpy(), weights[f'{name}.{role}'])
        with np.load(C.OUT / f'prediction-check-{step:06d}.npz') as predictions:
            for meta in C.network_rows(step):
                model = {role: weights[f'{meta["activation"]}.{role}'][meta['member']].astype(float)
                         for role in ['gate', 'out', 'bias', 'out_bias']}
                for sigma in [0., .35]:
                    x = assets['eval_bits'][:128].astype(float) @ assets['encoder'].T + sigma * assets['eval_noise'][:128].astype(float)
                    pred = C.numpy_forward(x, model, meta['activation'])
                    parity_error = max(parity_error, float(np.abs(pred - predictions[f'{meta["activation"]}_{sigma}'][meta['member']]).max()))
        for meta in C.network_rows(step):
            path = C.OUT / 'analysis' / (meta['key'] + '.json'); record = json.loads(path.read_text())
            assert record['dependencies'] == {**C.dependencies(), 'weights_sha256': C.sha(C.OUT / f'weights-{step:06d}.safetensors')}
            assert record['raw_sha256'] == C.sha(path.with_suffix('.npz'))
            target = C.targets(assets['recovery_bits'].astype(float), meta['task'])
            with np.load(path.with_suffix('.npz')) as raw:
                assert raw['response'].shape == (8, 4, 31, 31)
                assert np.isfinite(raw['response']).all() and raw['response'].min() >= 0
                assert raw['response'][..., 0, 0].max() < 1e-10
                axes = np.linalg.norm(raw['axis_vectors'], axis=-1)
                assert np.abs(axes[:, :, 0] - raw['response'][..., :, 0]).max() < 1e-10
                assert np.abs(axes[:, :, 1] - raw['response'][..., 0, :]).max() < 1e-10
                for level, sigma in enumerate(C.P['recovery']['noise_std']):
                    natural = raw['recovery_predictions'][level].astype(float)
                    clean = assets['recovery_bits'].astype(float) @ assets['encoder'].T
                    x = clean[None] + sigma * assets['recovery_noise']
                    scores = {'natural': natural, 'clipped_natural': np.clip(natural, 0, 1)}
                    scores.update({key: value.reshape(2, len(target), 8) for key, value in
                                   C.references(x.reshape(-1, 32), assets['encoder'], meta['task'], sigma).items()})
                    for name, prediction in scores.items():
                        expected = C.score(prediction, target)
                        row = next(r for r in record['recovery'] if r['sigma'] == sigma and r['map'] == name)
                        score_error = max(score_error, max(abs(row[key] - value) for key, value in expected.items()))
                model = A.model_from(meta); predict = lambda x: C.numpy_forward(x, model, meta['activation'])
                for index, case in enumerate(record['geometry']):
                    for background in case['backgrounds']:
                        attempted += 1; flagged += bool(background['fit']['all_flags'])
                    if case['state'] == [0, 0]:
                        refit = A.fit(raw['response'][index, 3]); old = case['backgrounds'][3]['fit']
                        assert refit['all_flags'] == old['all_flags']
                        if old['p'] is not None: fit_error = max(fit_error, abs(refit['p'] - old['p']))
                        primary.append({'network': meta, 'pair': case['pair']['name'], 'fit': old})
                        if meta['seed'] == 0:
                            measured = A.geometry_case(predict, assets, case['pair'], case['state'], 31)
                            grid_error = max(grid_error, float(np.abs(measured[0] - raw['response'][index]).max()))
                        if record['fine_geometry']:
                            finer = next(r for r in record['fine_geometry'] if r['pair'] == case['pair'])
                            for b in range(4):
                                coarse = case['backgrounds'][b]['fit']; fine_fit = finer['backgrounds'][b]['fit']
                                delta = fine_fit['p'] - coarse['p'] if coarse['p'] is not None and fine_fit['p'] is not None else None
                                fine.append({'network': meta, 'pair': case['pair']['name'], 'background': b,
                                             'delta_p': delta, 'jointly_unflagged': not (coarse['all_flags'] or fine_fit['all_flags'])})
            frames += 1
    assert frames == 180 and attempted == 5760 and len(fine) == 160
    assert parity_error < 3e-5 and score_error == 0 and grid_error < 1e-12 and fit_error < 1e-12
    result = {'all_assertions_passed': True, 'networks': 60, 'frames': frames, 'background_fits': attempted,
              'flagged_background_fits': flagged, 'NumPy_vs_saved_MPS_prediction_max_error': parity_error,
              'raw_prediction_score_reconstruction_max_error': score_error,
              'independent_geometry_recalculation_max_error': grid_error, 'canonical_raw_refit_max_error': fit_error,
              'fine_checks': len(fine), 'fine_checks_jointly_unflagged': sum(r['jointly_unflagged'] for r in fine),
              'fine_max_abs_delta_p_jointly_unflagged': max((abs(r['delta_p']) for r in fine if r['jointly_unflagged'] and r['delta_p'] is not None), default=None),
              'dependencies': C.dependencies()}
    C.write(C.OUT / 'fine-checks.json', fine); C.write(C.OUT / 'primary-fits.json', primary)
    C.write(C.OUT / 'quality-audit.json', result); print(json.dumps(C.BASE.clean(result), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', choices=['preflight', 'audit'], required=True)
    args = parser.parse_args(); preflight() if args.stage == 'preflight' else audit()
