"""Task, model and immutable inputs for the small AND/XOR pilot."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from scipy.special import erf, expit
import torch
import characterization_core as BASE

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / 'configs/toy-logic-v1.json').read_text())
OUT = Path(os.environ.get('QWENCH_LOGIC_OUT', ROOT / 'reports/toy-logic-v1')).resolve()
WORK = ROOT / 'work/toy-logic-v1' if OUT == ROOT / 'reports/toy-logic-v1' else OUT / 'training-state'
sha = BASE.sha
write = BASE.write
forward = BASE.forward
activation = BASE.activation


def members():
    return [(task, seed) for task in P['tasks'] for seed in P['seeds']]


def targets(bits, task):
    a, b = bits[..., 0::2], bits[..., 1::2]
    return a * b if task.startswith('and') else a + b - 2 * a * b


def initialize(device='cpu', dtype=torch.float32):
    initial = {}
    for seed in P['seeds']:
        rng = torch.Generator().manual_seed(seed)
        initial[seed] = {
            'gate': torch.randn(64, 32, generator=rng) / np.sqrt(32),
            'out': torch.randn(8, 64, generator=rng) / np.sqrt(64),
            'bias': torch.zeros(64), 'out_bias': torch.zeros(8)}
    groups = {}
    for name in P['activations']:
        groups[name] = {role: torch.stack([initial[seed][role].clone() for task, seed in members()])
                        .to(device=device, dtype=dtype).requires_grad_() for role in initial[0]}
    return groups, [value for group in groups.values() for value in group.values()]


def numpy_forward(x, model, name):
    z = x @ model['gate'].T + model['bias']
    if name == 'relu': h = np.maximum(z, 0)
    elif name == 'gelu': h = .5 * z * (1 + erf(z / np.sqrt(2)))
    elif name == 'silu': h = z * expit(z)
    elif name == 'leaky_relu': h = np.where(z >= 0, z, .1 * z)
    elif name == 'relu2': h = np.maximum(z, 0) ** 2
    else: raise ValueError(name)
    return h @ model['out'].T + model['out_bias']


def references(x, encoder, task, sigma):
    decoded = x @ encoder
    plugin = targets(np.clip(decoded, 0, 1), task)
    if sigma == 0:
        # These calls receive exact clean codewords; no noiseless off-manifold posterior is claimed.
        posterior_bits = (decoded > .5).astype(float)
    else:
        q = P['task']['feature_probability']
        posterior_bits = expit((2 * decoded - 1) / (2 * sigma ** 2) + np.log(q / (1 - q)))
    oracle = targets(posterior_bits, task)
    prior = .25 if task.startswith('and') else .5
    return {'plugin_reference': plugin, 'Bayes_oracle': oracle,
            'constant_prior': np.full_like(plugin, prior)}


def score(prediction, target):
    target = np.broadcast_to(target, prediction.shape)
    error = np.square(prediction - target)
    positive = target == 1; negative = target == 0
    pos = float(error[positive].mean()); neg = float(error[negative].mean())
    return {'all_mse': float(error.mean()), 'positive_mse': pos, 'negative_mse': neg,
            'balanced_mse': .5 * (pos + neg),
            'positive_miss_rate': float((prediction[positive] < .5).mean()),
            'negative_false_positive_rate': float((prediction[negative] >= .5).mean()),
            'positive_count': int(positive.sum()), 'negative_count': int(negative.sum())}


def source_dependencies():
    files = ['configs/toy-logic-v1.json', 'scripts/logic_core.py', 'scripts/train_logic.py',
             'scripts/check_logic.py', 'scripts/analyze_logic.py',
             'scripts/characterization_core.py', 'scripts/analyze_toy_cis.py',
             'configs/toy-characterization-v1.json', 'configs/toy-cis-v1.json',
             'reports/toy-cis-v1/prepared.json', 'vendor/fsec/scripts/lib/superellipse.py']
    return {path: sha(ROOT / path) for path in files}


def dependencies():
    return {'sources': source_dependencies(), 'assets_sha256': sha(OUT / 'assets.npz')}


def make_assets():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'assets.npz').exists(), 'Prepared assets already exist; refusing overwrite.'
    arrays = {'encoder': BASE.encoder()}
    for label, n, seed in [('train', P['training']['train_pool_size'], P['training']['data_seed']),
                           ('eval', P['training']['eval_size'], P['training']['eval_seed']),
                           ('recovery', P['recovery']['samples'], P['recovery']['data_seed'])]:
        arrays[label + '_bits'] = (np.random.default_rng(seed).random((n, 16)) < .5).astype(np.float32)
    arrays['eval_noise'] = np.random.default_rng(P['training']['eval_seed'] + 100).normal(size=(2048, 32)).astype(np.float32)
    arrays['recovery_noise'] = np.random.default_rng(P['recovery']['noise_seed']).normal(size=(2, P['recovery']['samples'], 32))
    rng = np.random.default_rng(P['geometry']['background_seed'])
    backgrounds = np.repeat((rng.random((1, 16)) < .5).astype(float), 4, axis=0)
    backgrounds[:, [1, 3]] = np.asarray(P['geometry']['states'])
    arrays['background_bits'] = backgrounds
    np.savez_compressed(OUT / 'assets.npz', **arrays)
    write(OUT / 'prepared.json', {'dependencies': dependencies(), 'networks': 60,
          'task_prior': {'and_positive': .25, 'xor_positive': .5},
          'geometry_comparison': P['geometry']['primary_comparison']})


def load_assets():
    with np.load(OUT / 'assets.npz') as raw: return {key: raw[key].copy() for key in raw.files}


def assert_seal():
    prepared = json.loads((OUT / 'prepared.json').read_text())
    assert prepared['dependencies'] == dependencies(), 'Frozen source/input changed after preparation.'


def network_rows(step):
    return [{'activation': name, 'task': task, 'seed': seed, 'member': member, 'step': step,
             'key': f'{name}_{task}_s{seed}_t{step:06d}'}
            for name in P['activations'] for member, (task, seed) in enumerate(members())]
