"""Independent preflight checks before the characterization's full training."""
import copy
import io
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import erf, expit, logsumexp
import torch
import characterization_core as C


def numpy_activation(z, name):
    if name == 'relu': return np.maximum(z, 0)
    if name == 'gelu': return .5 * z * (1 + erf(z / np.sqrt(2)))
    if name == 'silu': return z * expit(z)
    if name == 'leaky_relu': return np.where(z >= 0, z, .1 * z)
    if name == 'relu2': return np.maximum(z, 0) ** 2
    raise ValueError(name)


def numpy_forward(x, model, name):
    values = {key: value.detach().cpu().numpy() for key, value in model.items()}
    return numpy_activation(x @ values['gate'].T + values['bias'], name) @ values['out'].T + values['out_bias']


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(55199)
    groups, _ = C.initialize(dtype=torch.float64)
    checks = {}; initialization_match = True
    for name, group in groups.items():
        for role in group:
            assert torch.equal(group[role], groups['relu'][role])
            for task_index in range(4): assert torch.equal(group[role][:3], group[role][task_index*3:task_index*3+3])
        x = torch.tensor(rng.normal(size=(12, 7, 33)), dtype=torch.float64)
        y = torch.tensor(rng.normal(size=(12, 7, 16)), dtype=torch.float64)
        batch = {key: value.detach().clone().requires_grad_() for key, value in group.items()}
        opt = torch.optim.Adam(list(batch.values()), lr=.001, foreach=True)
        ((C.forward(x, batch, name) - y).square().mean((1, 2))).sum().backward()
        batch_grad = {key: value.grad.clone() for key, value in batch.items()}
        opt.step(); grad_error = 0.; update_error = 0.; derivative_error = 0.; numpy_error = 0.
        for index in range(12):
            model = {key: value[index].detach().clone().requires_grad_() for key, value in group.items()}
            single_opt = torch.optim.Adam(list(model.values()), lr=.001, foreach=True)
            (C.forward(x[index], model, name) - y[index]).square().mean().backward()
            for key, value in model.items(): grad_error = max(grad_error, float((value.grad - batch_grad[key][index]).abs().max()))
            single_opt.step()
            for key, value in model.items(): update_error = max(update_error, float((value - batch[key][index]).abs().max().detach()))
            expected = numpy_forward(x[index].numpy(), model, name)
            numpy_error = max(numpy_error, float(np.abs(expected - C.forward(x[index], model, name).detach().numpy()).max()))
            direction = torch.tensor(rng.normal(size=x[index].shape), dtype=torch.float64)
            _, actual = torch.autograd.functional.jvp(lambda v: C.forward(v, model, name), x[index], direction)
            z = x[index] @ model['gate'].T + model['bias']
            analytic = ((direction @ model['gate'].T) * C.derivative(z, name)) @ model['out'].T
            derivative_error = max(derivative_error, float((actual - analytic).abs().max().detach()))
        assert max(grad_error, update_error, numpy_error, derivative_error) < 1e-10
        checks[name] = {'batched_vs_separate_gradient_max_error': grad_error, 'Adam_update_max_error': update_error,
                        'numpy_forward_max_error': numpy_error, 'autograd_derivative_max_error': derivative_error}
    a = C.encoder(); assert np.abs(a.T @ a - np.eye(16)).max() < 1e-12
    bits = (rng.random((10, 16)) < .125).astype(float); contexts = rng.choice([-1., 1.], 10)
    inversion = []
    for task in C.P['tasks']:
        encoded = C.task_inputs(torch.tensor(bits), torch.tensor(contexts), torch.tensor(a), task).numpy()
        for sign in [-1, 1]:
            mask = contexts == sign
            inversion.append(float(np.abs(encoded[mask, :32] @ C.effective_encoder(a, task, sign) - bits[mask]).max()))
    assert max(inversion) < 1e-12
    all_bits = np.array(list(itertools.product([0., 1.], repeat=16)))
    counts = all_bits.sum(1); log_prior = counts * np.log(.125) + (16 - counts) * np.log(.875)
    bayes_errors = []
    for sign in [-1., 1.]:
        aa = C.effective_encoder(a, 'context_denoise', sign)
        observation = aa @ bits[0] + .35 * rng.normal(size=32)
        log_like = -np.square(observation - all_bits @ aa.T).sum(1) / (2 * .35 ** 2)
        weights = np.exp(log_prior + log_like - logsumexp(log_prior + log_like))
        enumerated = weights @ all_bits
        _, posterior = C.posterior_mean(observation[None], a, np.array([sign]), 'context_denoise', .35)
        bayes_errors.append(float(np.abs(enumerated - posterior[0]).max()))
    assert max(bayes_errors) < 1e-10
    mps_parity = {}; resume_error = None
    if torch.backends.mps.is_available():
        float_groups, _ = C.initialize('mps')
        xx = torch.tensor(rng.normal(size=(12, 8, 33)), dtype=torch.float32)
        for name, model in float_groups.items():
            cpu = {key: value.detach().cpu() for key, value in model.items()}
            err = float((C.forward(xx.to('mps'), model, name).cpu() - C.forward(xx, cpu, name)).abs().max().detach())
            assert err < 2e-5
            mps_parity[name] = err
        torch.mps.manual_seed(55198)
        model = float_groups['relu']; opt = torch.optim.Adam(list(model.values()), lr=.001, foreach=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100000)
        def step(active, optimizer, schedule):
            x = torch.randn(12, 8, 33, device='mps'); y = torch.randn(12, 8, 16, device='mps')
            optimizer.zero_grad(set_to_none=True); loss = (C.forward(x, active, 'relu') - y).square().mean((1, 2)).sum()
            loss.backward(); optimizer.step(); schedule.step(); return x.detach().cpu(), y.detach().cpu()
        for _ in range(3): step(model, opt, scheduler)
        stream = io.BytesIO()
        torch.save({'weights': {key: value.detach().cpu() for key, value in model.items()},
                    'optimizer': opt.state_dict(), 'schedule': scheduler.state_dict(), 'rng': torch.mps.get_rng_state()}, stream)
        xx, yy = step(model, opt, scheduler); expected = {key: value.detach().cpu().clone() for key, value in model.items()}
        stream.seek(0); state = torch.load(stream, map_location='cpu', weights_only=True)
        restored = {key: value.to('mps').requires_grad_() for key, value in state['weights'].items()}
        opt2 = torch.optim.Adam(list(restored.values()), lr=.001, foreach=True)
        opt2.load_state_dict(state['optimizer']); sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 100000)
        sched2.load_state_dict(state['schedule']); torch.mps.set_rng_state(state['rng'])
        xxx, yyy = step(restored, opt2, sched2); assert torch.equal(xx, xxx) and torch.equal(yy, yyy)
        resume_error = max(float((value.detach().cpu() - expected[key]).abs().max()) for key, value in restored.items())
        assert resume_error < 2e-6
    result = {'all_assertions_passed': True, 'gradient_update_forward_derivative_checks': checks,
              'exact_initialization_pairing': True, 'encoder_inversion_max_error': max(inversion),
              'Bayes_vs_full_65536_latent_likelihood_enumeration_max_error': max(bayes_errors),
              'MPS_CPU_forward_max_errors': mps_parity, 'MPS_checkpoint_resume_max_error': resume_error,
              'source_sha256': C.sha(__file__), 'core_sha256': C.sha(C.__file__)}
    C.write(C.OUT / 'preflight-checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
