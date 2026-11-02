"""Construct toy-network tensors from formulas, without training or checkpoints.

The constructors accept only architecture/task constants. No examples, targets,
optimizer, autograd, or trained weights are used to construct these tensors.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from safetensors.numpy import save_file


def axis_selector():
    """Exactly compute 50 of the 100 ReLU coordinates."""
    gate = np.eye(100)[:50]
    return {'gate': gate, 'out': gate.T.copy()}


def sparse_code(ones=5, gain=.9, seed=62002):
    """Positive code + centered algebraic inverse, with fixed isolated gain.

    Each feature activates `ones` of 50 neurons. Centering removes the common
    mode. The pseudoinverse is of this *constructed matrix*, not sampled data.
    Isolated negative inputs activate nothing. Positive inputs reconstruct a
    rescaled rank-49 projection, trading off cross-talk against coverage.
    """
    rng = np.random.default_rng(seed)
    gate = np.zeros((50, 100))
    for feature in range(100):
        gate[rng.choice(50, ones, replace=False), feature] = 1
    out = np.linalg.pinv(gate - gate.mean(axis=0))
    out *= (gain / np.diag(out @ gate))[:, None]
    return {'gate': gate, 'out': out}


def binary_decoder(encoder, context=False, denoise=False, sigma=.35, prior=.125):
    """Compile inverse coordinates and piecewise-linear thresholds into an MLP.

    Exactly the existing 33 -> 64 ReLU -> 16 biased architecture. For simple
    denoising, use a four-hinge sigmoid approximation. Context routing leaves
    two hinges per feature/branch, so use a ramp. Inactive context branches are
    shifted left by 8 latent units; their suppression is exact for clean bits,
    but not a mathematical guarantee for unbounded Gaussian noise.

    For overlapping codes this uses marginal dual-coordinate variance, ignoring
    correlations. It is deliberately not the exact coupled Bayes estimator.
    """
    gate = np.zeros((64, 33)); out = np.zeros((16, 64))
    bias = np.zeros(64); out_bias = np.zeros(16)
    neuron = 0
    for sign in ([-1, 1] if context else [1]):
        a = encoder[:, np.roll(np.arange(16), 8)] if context and sign < 0 else encoder
        dual = a @ np.linalg.inv(a.T @ a)
        for feature in range(16):
            variance = sigma**2 * np.sum(dual[:, feature]**2)
            if denoise:
                midpoint = .5 + variance * np.log((1-prior)/prior)
                if context:
                    knots = midpoint + variance * np.array([-2., 2.])
                    values = np.array([0., 1.])
                else:
                    knots = midpoint + variance * np.array([-3., -1., 1., 3.])
                    values = np.array([0., .25, .75, 1.])
            else:
                knots = np.array([0., 1.]); values = np.array([0., 1.])
            slopes = np.diff(values) / np.diff(knots)
            coefficients = np.diff(np.r_[0., slopes, 0.])
            for knot, coefficient in zip(knots, coefficients):
                gate[neuron, :32] = dual[:, feature]
                bias[neuron] = -knot
                if context:
                    gate[neuron, -1] = 4 * sign
                    bias[neuron] -= 4
                out[feature, neuron] = coefficient
                neuron += 1
    assert neuron <= 64
    return {'gate': gate, 'out': out, 'bias': bias, 'out_bias': out_bias}


def build_models():
    # Import only task definitions; constructors never read saved checkpoints.
    import characterization_core as C
    models = {'axis50': axis_selector(), 'code5_gain09': sparse_code(5),
              'code10_gain09': sparse_code(10)}
    metadata = {name: {'track': 'bottleneck', 'activation': 'relu', 'width': 50,
                      'parameters': 10000} for name in models}
    base = C.encoder()
    for family in ['orthogonal', 'coupled']:
        a = base.copy()
        if family == 'coupled':
            a[:, 1::2] = .6 * base[:, 0::2] + .8 * base[:, 1::2]
        for task in C.P['tasks']:
            name = f'{family}_{task}'
            models[name] = binary_decoder(a, task.startswith('context'), task.endswith('denoise'))
            metadata[name] = {'track': 'binary', 'family': family, 'task': task,
                              'activation': 'relu', 'width': 64, 'parameters': 3216}
    return {name: {role: np.ascontiguousarray(value, dtype=np.float32)
                   for role, value in model.items()} for name, model in models.items()}, metadata


def save_models(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    models, metadata = build_models()
    save_file({f'{name}.{role}': value for name, model in models.items()
               for role, value in model.items()}, str(destination / 'weights.safetensors'))
    (destination / 'models.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return models, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten.')
    args = parser.parse_args()
    save_models(args.out)
    print(f'Wrote 11 constructed models to {args.out}')
