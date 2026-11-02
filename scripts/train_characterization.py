"""Train the frozen 60-network coded-bit characterization on native MPS."""
import argparse
import json
import time
import subprocess

import numpy as np
import torch
from safetensors.torch import save_file
import characterization_core as C


def dependencies():
    return {'source_sha256': C.sha(__file__), 'core_source_sha256': C.sha(C.__file__),
            'protocol_sha256': C.sha(C.ROOT / 'configs/toy-characterization-v1.json'),
            'assets_sha256': C.sha(C.OUT / 'assets.npz')}


@torch.inference_mode()
def evaluate(groups, assets, step):
    device = next(iter(groups.values()))['gate'].device
    bits = torch.tensor(assets['eval_bits'], device=device)
    context = torch.tensor(assets['eval_context'], device=device)
    a = torch.tensor(assets['encoder'], device=device, dtype=torch.float32)
    fixed_noise = torch.tensor(assets['eval_noise'], device=device)
    bases = torch.stack([C.task_inputs(bits, context, a, task) for task in C.P['tasks']]).repeat_interleave(3, 0)
    target = bits[None]; records = []; parity = {}
    for name, model in groups.items():
        predictions = {}
        for sigma in [0., C.P['task']['training_noise_std']]:
            x = bases.clone(); x[..., :32] += sigma * fixed_noise
            predictions[sigma] = C.forward(x, model, name)
        for index, (task, seed) in enumerate(C.members()):
            row = {'id': f'{name}_{task}_s{seed}', 'activation': name, 'task': task, 'seed': seed, 'step': step}
            for sigma, pred in predictions.items():
                error = (pred[index] - bits).square()
                row['clean' if sigma == 0 else 'noise035'] = {
                    'mse': float(error.mean()), 'active_mse': float(error[bits > 0].mean()),
                    'inactive_mse': float(error[bits == 0].mean()),
                    'per_feature_active_mse': (error * bits).sum(0).div(bits.sum(0)).cpu().tolist(),
                    'per_feature_inactive_mse': (error * (1 - bits)).sum(0).div((1 - bits).sum(0)).cpu().tolist()}
            single = {role: value[index] for role, value in model.items()}
            all_gains = []; all_argmax = []
            for sign in [-1, 1]:
                aa = torch.tensor(C.effective_encoder(assets['encoder'], task, sign), device=device, dtype=torch.float32)
                probes = torch.cat([aa.T, torch.full((16, 1), float(sign), device=device)], -1)
                zero = torch.zeros(1, 33, device=device); zero[:, -1] = sign
                response = C.forward(probes, single, name) - C.forward(zero, single, name)
                all_gains.append(response.diagonal()); all_argmax.append(response.argmax(-1) == torch.arange(16, device=device))
            gains = torch.stack(all_gains).min(0).values.cpu().numpy()
            argmax = torch.stack(all_argmax).all(0).cpu().numpy()
            eligible = (np.asarray(row['clean']['per_feature_active_mse']) < .1) & (np.asarray(row['clean']['per_feature_inactive_mse']) < .05) & (gains > .5) & argmax
            row.update(competent_features=int(eligible.sum()), competent_indices=np.where(eligible)[0].tolist(),
                       min_context_single_feature_gains=gains.tolist())
            records.append(row)
        parity[name] = predictions[0.][:, :128].cpu().numpy()
    C.write(C.OUT / f'eval-{step:06d}.json', {'step': step, 'networks': records, 'dependencies': dependencies()})
    np.savez_compressed(C.OUT / f'prediction-check-{step:06d}.npz', **parity)
    return records


def main(resume=False, benchmark=0):
    assert torch.backends.mps.is_available(), 'Native MPS is required for this training protocol'
    torch.set_num_threads(2); C.WORK.mkdir(parents=True, exist_ok=True)
    if not (C.OUT / 'assets.npz').exists(): C.make_assets()
    with np.load(C.OUT / 'assets.npz') as raw: assets = {key: raw[key].copy() for key in raw.files}
    groups, params = C.initialize('mps')
    cfg = C.P['training']; optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'], betas=tuple(cfg['betas']), eps=cfg['eps'], foreach=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['steps'])
    bits = torch.tensor(assets['train_bits'], device='mps'); context = torch.tensor(assets['train_context'], device='mps')
    a = torch.tensor(assets['encoder'], device='mps', dtype=torch.float32)
    bases = torch.stack([C.task_inputs(bits, context, a, task) for task in C.P['tasks']])
    noise_mask = torch.tensor([cfg_task.endswith('denoise') for cfg_task in C.P['tasks']], device='mps')[:, None, None]
    checkpoint = C.WORK / 'resume.pt'; start = 0; prior_seconds = 0.
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['dependencies'] == dependencies()
        for name, group in groups.items():
            for role, value in group.items(): value.data.copy_(state['weights'][name][role].to('mps'))
        optimizer.load_state_dict(state['optimizer']); schedule.load_state_dict(state['schedule'])
        torch.mps.set_rng_state(state['rng']); start = state['step']; prior_seconds = state['seconds']
        path = C.OUT / 'training.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        discarded = [row for row in records if row['step'] > start]
        if discarded:
            C.write(C.WORK / f'discarded-after-{start}.json', discarded)
            path.write_text(''.join(json.dumps(row) + '\n' for row in records if row['step'] <= start))
    else:
        if not benchmark:
            assert not checkpoint.exists() and not (C.OUT / 'training.jsonl').exists(), 'Use --resume; refuse overwrite'
        torch.mps.manual_seed(cfg['minibatch_seed'])
    if not benchmark and not resume:
        C.write(C.OUT / 'training-input-seal.json', {**dependencies(), 'torch': torch.__version__, 'device': 'mps',
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.ROOT, text=True).strip()})
        save_file({f'{name}.{role}': value.detach().cpu().contiguous() for name, group in groups.items() for role, value in group.items()}, str(C.OUT / 'weights-000000.safetensors'))
        evaluate(groups, assets, 0)
    started = time.perf_counter(); last = started
    stop = benchmark if benchmark else cfg['steps']
    for step in range(start, stop):
        index = torch.randint(len(bits), (cfg['batch_size'],), device='mps')
        noise = torch.randn(cfg['batch_size'], 32, device='mps')
        x = bases[:, index].clone(); x[..., :32] += noise_mask * noise * C.P['task']['training_noise_std']
        x = x.repeat_interleave(len(C.P['seeds']), 0); target = bits[index]
        optimizer.zero_grad(set_to_none=True)
        losses = torch.stack([(C.forward(x, group, name) - target).square().mean((1, 2)) for name, group in groups.items()])
        losses.sum().backward(); optimizer.step(); schedule.step(); completed = step + 1
        if completed == 1 or completed % cfg['log_every'] == 0 or completed in cfg['checkpoints']:
            values = losses.detach().cpu().numpy(); assert np.isfinite(values).all()
            row = {'step': completed, 'seconds': prior_seconds + time.perf_counter() - started,
                   'lr': schedule.get_last_lr()[0], 'losses': {name: values[i].tolist() for i, name in enumerate(groups)}}
            if not benchmark:
                with (C.OUT / 'training.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                C.write(C.OUT / 'progress.json', row)
            print('TRAIN', completed, 'seconds', round(row['seconds'], 1), 'median losses', {name: round(float(np.median(v)), 6) for name, v in row['losses'].items()}, flush=True)
        if not benchmark and completed in cfg['checkpoints']:
            save_file({f'{name}.{role}': value.detach().cpu().contiguous() for name, group in groups.items() for role, value in group.items()}, str(C.OUT / f'weights-{completed:06d}.safetensors'))
            records = evaluate(groups, assets, completed)
            print('EVAL', completed, 'competent counts', {name: [row['competent_features'] for row in records if row['activation'] == name] for name in groups}, flush=True)
        if not benchmark and (completed % cfg['resume_every'] == 0 or completed == cfg['steps']):
            weights = {name: {role: value.detach().cpu() for role, value in group.items()} for name, group in groups.items()}
            state = {'weights': weights, 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                     'rng': torch.mps.get_rng_state(), 'step': completed, 'dependencies': dependencies(),
                     'seconds': prior_seconds + time.perf_counter() - started}
            temp = checkpoint.with_suffix('.tmp'); torch.save(state, temp); temp.replace(checkpoint)
    torch.mps.synchronize()
    if benchmark:
        print('BENCHMARK', benchmark, 'steps in', round(time.perf_counter() - started, 3), 'seconds', flush=True)
    else:
        C.write(C.OUT / 'TRAINING-COMPLETE.json', {'steps': cfg['steps'], 'networks': 60, 'dependencies': dependencies(),
                'seconds': prior_seconds + time.perf_counter() - started, 'final_rng_sha256': __import__('hashlib').sha256(torch.mps.get_rng_state().numpy().tobytes()).hexdigest()})
        print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', type=int, default=0)
    args = parser.parse_args(); main(args.resume, args.benchmark)
