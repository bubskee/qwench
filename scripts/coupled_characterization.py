"""Paired overlap-code extension using the unchanged original trainer/analyzer.

The adapter changes assets, the exact posterior, the dual-basis separable null,
and provenance paths. Activations, optimizer, architecture and update loop are
the same code as the orthogonal-code study; no activation/weight swaps occur.
"""
from pathlib import Path
import argparse
import itertools
import json
import sys

import numpy as np
from scipy.special import logsumexp
import torch
import characterization_core as B

ROOT = B.ROOT
P = json.loads((ROOT / 'configs/toy-characterization-coupled-v1.json').read_text())
OUT = ROOT / 'reports/toy-characterization-coupled-v1'
WORK = ROOT / 'work/toy-characterization-coupled-v1'
sha = B.sha; write = B.write; clean = B.clean
members = B.members; activation = B.activation; derivative = B.derivative
forward = B.forward; initialize = B.initialize
effective_encoder = B.effective_encoder; task_inputs = B.task_inputs


def encoder():
    a = B.encoder(); mixed = a.copy()
    mixed[:, 1::2] = .6 * a[:, 0::2] + .8 * a[:, 1::2]
    return mixed


def make_assets():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'assets.npz').exists()
    with np.load(B.OUT / 'assets.npz') as source:
        arrays = {key: source[key].copy() for key in source.files}
    arrays['encoder'] = encoder()
    np.savez_compressed(OUT / 'assets.npz', **arrays)
    write(OUT / 'prepared.json', {'assets_sha256': sha(OUT / 'assets.npz'),
          'parent_assets_sha256': sha(B.OUT / 'assets.npz'), 'protocol_sha256': sha(ROOT / 'configs' / (P['version'] + '.json')),
          'runtime_sha256': sha(__file__), 'base_core_sha256': sha(B.__file__),
          'networks': 60, 'pairing': 'All parent latent/context/noise arrays copied exactly; only encoder changes.'})


def posterior_mean(observation, a, context, task, sigma):
    references = np.empty((len(observation), 16)); oracle = np.empty_like(references)
    states = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    counts = states.sum(1); q = P['task']['feature_probability']
    log_prior = counts * np.log(q) + (2 - counts) * np.log(1 - q)
    for sign in [-1, 1]:
        mask = context == sign
        if not mask.any(): continue
        aa = effective_encoder(a, task, sign); gram = aa.T @ aa
        matched = observation[mask] @ aa
        references[mask] = np.clip(matched @ np.linalg.inv(gram), 0, 1)
        if sigma == 0:
            oracle[mask] = references[mask]; continue
        predictions = []
        for pair in range(8):
            g = gram[2*pair:2*pair+2, 2*pair:2*pair+2]
            energy = np.einsum('si,ij,sj->s', states, g, states)
            logit = (matched[:, 2*pair:2*pair+2] @ states.T - .5 * energy) / sigma**2 + log_prior
            probabilities = np.exp(logit - logsumexp(logit, axis=1)[:, None])
            predictions.append(probabilities @ states)
        oracle[mask] = np.concatenate(predictions, axis=1)
    return references, oracle


def separable(x, model, meta, a, outputs=None):
    indices = list(range(16)) if outputs is None else list(outputs)
    result = torch.empty((len(x), len(indices)), dtype=torch.float64)
    for sign in [-1., 1.]:
        mask = x[:, -1] == sign
        if not mask.any(): continue
        aa = torch.tensor(effective_encoder(a, meta['task'], sign), dtype=torch.float64)
        dual = aa @ torch.linalg.inv(aa.T @ aa)
        u = (x[mask, :32] @ dual)[:, indices]
        projections = (model['gate'][:, :32] @ aa)[:, indices].T
        offset = model['bias'] + sign * model['gate'][:, -1]
        h = activation(u[..., None] * projections[None] + offset[None, None], meta['activation'])
        result[mask] = (h * model['out'][indices][None]).sum(-1) + model['out_bias'][indices]
    return result


def training_dependencies():
    import train_characterization as T
    return {'source_sha256': sha(T.__file__), 'core_source_sha256': sha(__file__),
            'base_core_sha256': sha(B.__file__),
            'protocol_sha256': sha(ROOT / 'configs' / (P['version'] + '.json')),
            'assets_sha256': sha(OUT / 'assets.npz')}


def bind_analysis():
    import analyze_characterization as A
    A.C = sys.modules[__name__]; A.AN = OUT / 'analysis'; A.separable = separable
    return A


def prepare_analysis():
    analyzer = bind_analysis()
    analyzer.prepare()
    path = OUT / 'analysis-plan.json'; plan = json.loads(path.read_text())
    assert all(row['track'] == 'new' for row in plan['networks'])
    for row in plan['networks']:
        for name in ['weights', 'evaluation']:
            row[name] = row[name].replace('toy-characterization-v1/', P['version'] + '/')
    plan['protocol_sha256'] = sha(ROOT / 'configs' / (P['version'] + '.json'))
    plan['runtime_sha256'] = sha(__file__)
    write(path, plan)


def check():
    for key in ['activations', 'leaky_slope', 'seeds', 'tasks', 'network', 'training']:
        assert P[key] == B.P[key], f'Paired training setting differs: {key}'
    rng = np.random.default_rng(55231); a = encoder(); gram = a.T @ a
    expected = np.eye(16)
    for i in range(0, 16, 2): expected[i, i+1] = expected[i+1, i] = .6
    assert np.max(np.abs(gram - expected)) < 1e-12
    all_bits = np.array(list(itertools.product([0., 1.], repeat=16)))
    counts = all_bits.sum(1); q = P['task']['feature_probability']
    log_prior = counts * np.log(q) + (16 - counts) * np.log(1 - q)
    errors = []
    for sign in [-1., 1.]:
        aa = effective_encoder(a, 'context_denoise', sign)
        z = (rng.random(16) < q).astype(float)
        observation = aa @ z + .35 * rng.normal(size=32)
        log_weights = log_prior - np.square(observation - all_bits @ aa.T).sum(1) / (2 * .35**2)
        enumeration = np.exp(log_weights - logsumexp(log_weights)) @ all_bits
        _, actual = posterior_mean(observation[None], a, np.array([sign]), 'context_denoise', .35)
        errors.append(float(np.abs(actual[0] - enumeration).max()))
    assert max(errors) < 1e-10
    observations = np.zeros((2, 16)); observations[:, 0] = .4; observations[1, 1] = .8
    _, posterior = posterior_mean(observations @ a.T, a, np.ones(2), 'linear_denoise', .35)
    coupling = float(abs(posterior[0, 0] - posterior[1, 0]))
    assert coupling > 1e-4, 'Posterior must use neighboring decoded estimates'
    # Exact positive/negative single-feature response matching for every activation and both contexts.
    groups, _ = initialize(dtype=torch.float64); null_errors = []
    for name, group in groups.items():
        model = {role: value[0].detach() for role, value in group.items()}
        for sign in [-1., 1.]:
            aa = effective_encoder(a, 'context_denoise', sign)
            for j in [0, 1, 9]:
                values = np.array([-.5, 0., .25, 1., 1.5])
                x = torch.tensor(np.c_[values[:, None] * aa[:, j], np.full(5, sign)])
                natural = forward(x, model, name)[:, j]
                null = separable(x, model, {'task': 'context_denoise', 'activation': name}, a)[:, j]
                null_errors.append(float((natural - null).abs().max()))
    assert max(null_errors) < 1e-10
    with np.load(B.OUT / 'assets.npz') as old, np.load(OUT / 'assets.npz') as new:
        assert all(np.array_equal(old[key], new[key]) for key in old.files if key != 'encoder')
    write(OUT / 'preflight-checks.json', {'all_assertions_passed': True,
          'exact_parent_data_noise_pairing': True, 'encoder_Gram_error': float(np.abs(gram - expected).max()),
          'coupled_posterior_vs_65536_state_enumeration_max_error': max(errors),
          'neighbor_dependence_witness': coupling, 'separable_isolated_curve_max_error': max(null_errors),
          'same_trainer_and_initialization_as_parent': True, 'runtime_sha256': sha(__file__), 'base_core_sha256': sha(B.__file__)})
    print('PASS paired assets, exact coupled posterior, genuine neighbor dependence, and dual-basis null.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'check', 'train', 'analysis-prepare', 'measure'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare': make_assets()
    elif args.stage == 'check': check()
    elif args.stage == 'train':
        import train_characterization as T
        T.C = sys.modules[__name__]; T.dependencies = training_dependencies
        T.main(resume=args.resume)
    elif args.stage == 'analysis-prepare': prepare_analysis()
    else:
        with torch.inference_mode(): bind_analysis().measure('new')


if __name__ == '__main__': main()
