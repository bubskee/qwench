"""Shared task/model definitions for the controlled toy characterization."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / 'configs/toy-characterization-v1.json').read_text())
OUT = ROOT / 'reports/toy-characterization-v1'
WORK = ROOT / 'work/toy-characterization-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n'); temp.replace(path)


def members(): return [(task, seed) for task in P['tasks'] for seed in P['seeds']]


def activation(z, name):
    if name == 'relu': return z.relu()
    if name == 'gelu': return F.gelu(z, approximate='none')
    if name == 'silu': return F.silu(z)
    if name == 'leaky_relu': return F.leaky_relu(z, P['leaky_slope'])
    if name == 'relu2': return z.relu().square()
    raise ValueError(name)


def derivative(z, name):
    if name == 'relu': return (z > 0).to(z.dtype)
    if name == 'gelu': return .5 * (1 + torch.erf(z / np.sqrt(2))) + z * torch.exp(-z.square() / 2) / np.sqrt(2 * np.pi)
    if name == 'silu':
        s = z.sigmoid(); return s + z * s * (1 - s)
    if name == 'leaky_relu': return torch.where(z >= 0, torch.ones_like(z), torch.full_like(z, P['leaky_slope']))
    if name == 'relu2': return 2 * z.relu()
    raise ValueError(name)


def forward(x, model, name):
    h = activation(x @ model['gate'].transpose(-1, -2) + model['bias'].unsqueeze(-2), name)
    return h @ model['out'].transpose(-1, -2) + model['out_bias'].unsqueeze(-2)


def initialize(device='cpu', dtype=torch.float32):
    sizes = P['network']; initial = {}
    for seed in P['seeds']:
        rng = torch.Generator().manual_seed(seed)
        initial[seed] = {'gate': torch.randn(sizes['hidden'], sizes['inputs'], generator=rng) / np.sqrt(sizes['inputs']),
                         'out': torch.randn(sizes['outputs'], sizes['hidden'], generator=rng) / np.sqrt(sizes['hidden']),
                         'bias': torch.zeros(sizes['hidden']), 'out_bias': torch.zeros(sizes['outputs'])}
    groups = {}; params = []
    for name in P['activations']:
        group = {role: torch.stack([initial[seed][role].clone() for task, seed in members()]).to(device=device, dtype=dtype).requires_grad_()
                 for role in initial[0]}
        groups[name] = group; params.extend(group.values())
    return groups, params


def encoder():
    rng = np.random.default_rng(P['task']['encoder_seed'])
    q, r = np.linalg.qr(rng.normal(size=(32, 16)))
    return q * np.where(np.diag(r) >= 0, 1., -1.)


def effective_encoder(a, task, context):
    return a[:, np.roll(np.arange(16), 8)] if task.startswith('context') and context < 0 else a


def task_inputs(bits, context, a, task):
    linear = bits @ a.T
    if task.startswith('context'):
        alternate = bits @ a[:, np.roll(np.arange(16), 8)].T
        linear = torch.where(context[:, None] < 0, alternate, linear)
    return torch.cat([linear, context[:, None]], dim=-1)


def make_assets(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'assets.npz'
    if path.exists(): raise ValueError('Refusing to replace prepared assets')
    arrays = {'encoder': encoder()}
    for label, n, seed in [('train', P['training']['train_pool_size'], P['training']['data_seed']),
                           ('eval', P['training']['eval_size'], P['training']['eval_seed']),
                           ('recovery', P['recovery']['samples'], P['recovery']['data_seed'])]:
        rng = np.random.default_rng(seed)
        arrays[label + '_bits'] = (rng.random((n, 16)) < P['task']['feature_probability']).astype(np.float32)
        arrays[label + '_context'] = rng.choice([-1., 1.], size=n).astype(np.float32)
    rng = np.random.default_rng(P['training']['eval_seed'] + 100)
    arrays['eval_noise'] = rng.normal(size=(P['training']['eval_size'], 32)).astype(np.float32)
    arrays['recovery_noise'] = np.random.default_rng(P['recovery']['noise_seed']).normal(size=(2, P['recovery']['samples'], 32))
    rng = np.random.default_rng(P['geometry']['background_seed'])
    arrays['background_bits'] = (rng.random((4, 16)) < P['task']['feature_probability']).astype(float)
    arrays['background_context'] = np.asarray(P['geometry']['background_contexts'])
    rng = np.random.default_rng(P['geometry']['selection_seed'])
    random = rng.normal(size=(2, 32)); random[0] /= np.linalg.norm(random[0])
    random[1] -= random[1] @ random[0] * random[0]; random[1] /= np.linalg.norm(random[1])
    arrays['random_observation_pair'] = random
    np.savez_compressed(path, **arrays)
    write(out / 'prepared.json', {'protocol_sha256': sha(ROOT / 'configs/toy-characterization-v1.json'),
          'core_source_sha256': sha(__file__), 'assets_sha256': sha(path), 'members': members(),
          'networks': len(P['activations']) * len(members()), 'known_encoder_max_orthogonality_error': float(np.abs(arrays['encoder'].T @ arrays['encoder'] - np.eye(16)).max())})


def posterior_mean(observation, a, context, task, sigma):
    decoded = np.empty((len(observation), 16))
    for sign in [-1, 1]:
        mask = context == sign
        decoded[mask] = observation[mask] @ effective_encoder(a, task, sign)
    if sigma == 0: return np.clip(decoded, 0, 1), np.clip(decoded, 0, 1)
    log_prior = np.log(P['task']['feature_probability'] / (1 - P['task']['feature_probability']))
    logit = (2 * decoded - 1) / (2 * sigma * sigma) + log_prior
    posterior = np.exp(-np.logaddexp(0, -logit))
    return np.clip(decoded, 0, 1), posterior
