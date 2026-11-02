"""Fresh, paired AND/XOR training on native Apple MPS; no git/remote writes."""
import argparse
import json
import time

import numpy as np
from safetensors.torch import save_file
import torch
import logic_core as C


@torch.inference_mode()
def evaluate(groups, assets, step):
    device = next(iter(groups.values()))['gate'].device
    bits = torch.tensor(assets['eval_bits'], device=device)
    x = bits @ torch.tensor(assets['encoder'].T, device=device, dtype=torch.float32)
    noise = torch.tensor(assets['eval_noise'], device=device)
    rows = []; parity = {}
    for name, group in groups.items():
        predictions = {}
        for sigma in [0., .35]:
            pred = C.forward(x + sigma * noise, group, name).cpu().numpy()
            predictions[sigma] = pred
            parity[f'{name}_{sigma}'] = pred[:, :128]
        for member, (task, seed) in enumerate(C.members()):
            target = C.targets(assets['eval_bits'], task)
            rows.append({'activation': name, 'task': task, 'seed': seed, 'member': member,
                         'step': step, 'clean': C.score(predictions[0.][member], target),
                         'noise035': C.score(predictions[.35][member], target)})
    C.write(C.OUT / f'eval-{step:06d}.json', {'step': step, 'networks': rows, 'dependencies': C.dependencies()})
    np.savez_compressed(C.OUT / f'prediction-check-{step:06d}.npz', **parity)
    return rows


def main(resume=False, benchmark=0):
    C.assert_seal()
    assert torch.backends.mps.is_available(), 'Training requires Apple MPS.'
    torch.set_num_threads(2); C.WORK.mkdir(parents=True, exist_ok=True)
    assets = C.load_assets(); cfg = C.P['training']
    groups, params = C.initialize('mps')
    optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                                 eps=cfg['eps'], foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['steps'])
    bits = torch.tensor(assets['train_bits'], device='mps')
    observations = bits @ torch.tensor(assets['encoder'].T, device='mps', dtype=torch.float32)
    labels = torch.stack([C.targets(bits, task) for task, seed in C.members()])
    noise_mask = torch.tensor([task.endswith('denoise') for task, seed in C.members()], device='mps')[:, None, None]
    checkpoint = C.WORK / 'resume.pt'; start = 0; prior_seconds = 0.
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['dependencies'] == C.dependencies()
        for name, group in groups.items():
            for role, value in group.items(): value.data.copy_(state['weights'][name][role].to('mps'))
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.mps.set_rng_state(state['rng']); start = state['step']; prior_seconds = state['seconds']
        log = C.OUT / 'training.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        late = [row for row in rows if row['step'] > start]
        if late:
            C.write(C.WORK / f'discarded-after-{start}.json', late)
            log.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['step'] <= start))
    else:
        if not benchmark:
            assert not checkpoint.exists() and not (C.OUT / 'training.jsonl').exists(), 'Use --resume or a fresh output.'
        torch.mps.manual_seed(cfg['minibatch_seed'])
    if not benchmark and not resume:
        C.write(C.OUT / 'training-input-seal.json', {'dependencies': C.dependencies(),
                'torch': torch.__version__, 'numpy': np.__version__, 'device': 'mps', 'fresh_weights': True})
        save_file({f'{name}.{role}': value.detach().cpu().contiguous() for name, group in groups.items()
                   for role, value in group.items()}, str(C.OUT / 'weights-000000.safetensors'))
        evaluate(groups, assets, 0)
    started = time.perf_counter()
    for step in range(start, benchmark or cfg['steps']):
        index = torch.randint(len(bits), (cfg['batch_size'],), device='mps')
        noise = torch.randn(cfg['batch_size'], 32, device='mps')
        x = observations[index][None] + noise_mask * noise * .35
        target = labels[:, index]
        optimizer.zero_grad(set_to_none=True)
        losses = torch.stack([(C.forward(x, group, name) - target).square().mean((1, 2))
                              for name, group in groups.items()])
        losses.sum().backward(); optimizer.step(); scheduler.step(); completed = step + 1
        if completed == 1 or completed % cfg['log_every'] == 0:
            values = losses.detach().cpu().numpy(); assert np.isfinite(values).all()
            row = {'step': completed, 'seconds': prior_seconds + time.perf_counter() - started,
                   'lr': scheduler.get_last_lr()[0], 'losses': {name: values[i].tolist() for i, name in enumerate(groups)}}
            if not benchmark:
                with (C.OUT / 'training.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                C.write(C.OUT / 'progress.json', row)
            print('TRAIN', completed, 'seconds', round(row['seconds'], 1),
                  'median losses', {name: round(float(np.median(v)), 6) for name, v in row['losses'].items()}, flush=True)
        if not benchmark and completed in cfg['checkpoints']:
            save_file({f'{name}.{role}': value.detach().cpu().contiguous() for name, group in groups.items()
                       for role, value in group.items()}, str(C.OUT / f'weights-{completed:06d}.safetensors'))
            rows = evaluate(groups, assets, completed)
            print('EVAL', completed, 'median clean MSE', round(float(np.median([r['clean']['all_mse'] for r in rows])), 6), flush=True)
        if not benchmark and completed % cfg['resume_every'] == 0:
            state = {'weights': {name: {role: value.detach().cpu() for role, value in group.items()}
                                 for name, group in groups.items()}, 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'rng': torch.mps.get_rng_state(),
                     'step': completed, 'seconds': prior_seconds + time.perf_counter() - started,
                     'dependencies': C.dependencies()}
            temporary = checkpoint.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(checkpoint)
    torch.mps.synchronize()
    seconds = prior_seconds + time.perf_counter() - started
    if benchmark:
        print('BENCHMARK', benchmark, 'steps', round(seconds, 3), 'seconds', flush=True)
    else:
        C.write(C.OUT / 'TRAINING-COMPLETE.json', {'networks': 60, 'steps': cfg['steps'], 'seconds': seconds,
                'dependencies': C.dependencies(), 'checkpoint_sha256': C.sha(checkpoint)})
        print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--benchmark', type=int, default=0)
    args = parser.parse_args(); main(args.resume, args.benchmark)
