"""Fresh paired evaluation of formula-written weights and frozen trained models."""
from pathlib import Path
import argparse
import csv
import json
import subprocess

import numpy as np
import torch
from safetensors.numpy import load_file

import characterization_core as C
import coupled_characterization as K
from write_toy_weights import save_models

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/toy-handwritten-v1.json'
P = json.loads(PROTOCOL.read_text())


def predict(x, model, activation):
    tensors = {k: torch.from_numpy(v) for k, v in model.items()}
    tensors.setdefault('bias', torch.zeros(model['gate'].shape[0]))
    tensors.setdefault('out_bias', torch.zeros(model['out'].shape[0]))
    return C.forward(torch.from_numpy(x), tensors, activation).numpy()


def bottleneck_metrics(x, prediction, model, activation):
    error = np.square(prediction.astype(float) - np.maximum(x, 0))
    positive = x > 0; negative = x < 0; inactive = x == 0
    per = (error * positive).sum(0) / positive.sum(0)
    probes = predict(np.eye(100, dtype=np.float32), model, activation)
    gains = np.diag(probes)
    competent = (per < .05) & (gains >= .25) & (probes.argmax(1) == np.arange(100))
    h = C.activation(torch.from_numpy(model['gate'].T.copy()), activation).numpy()
    contribution = np.abs(h * model['out'])
    feature_pr = contribution.sum(1)**2 / np.maximum((contribution**2).sum(1), 1e-30)
    supported = contribution * competent[:, None]
    neuron_pr = supported.sum(0)**2 / np.maximum((supported**2).sum(0), 1e-30)
    active_neurons = neuron_pr > 0
    count = (x != 0).sum(1)
    row = {'all_mse': float(error.mean()), 'fourth_power_loss': float(np.square(error).mean()),
           'positive_mse': float(per.mean()), 'competent_features': int(competent.sum()),
           'inactive_rmse': float(np.sqrt(error[inactive].mean())),
           'negative_rmse': float(np.sqrt(error[negative].mean())),
           'inactive_false_positive_rate': float((prediction[inactive] > .1).mean()),
           'isolated_active_mse': float(error[positive & (count == 1)[:, None]].mean()),
           'multi_input_active_mse': float(error[positive & (count >= 2)[:, None]].mean()),
           'mean_competent_feature_participation': float(feature_pr[competent].mean()) if competent.any() else None,
           'mean_neuron_feature_participation': float(neuron_pr[active_neurons].mean()) if active_neurons.any() else None,
           'per_feature_positive_mse': per.tolist(), 'single_feature_gains': gains.tolist(),
           'competent_feature_indices': np.flatnonzero(competent).tolist()}
    for label, mask in [('positive', positive), ('inactive', inactive), ('negative', negative)]:
        row[label + '_squared_error_sum'] = float(error[mask].sum())
        row[label + '_count'] = int(mask.sum())
    return row


def binary_metrics(target, prediction):
    error = np.square(prediction.astype(float) - target)
    row = {'all_mse': float(error.mean()), 'out_of_range_fraction': float(((prediction < 0) | (prediction > 1)).mean())}
    for label, mask in [('active', target > 0), ('inactive', target == 0)]:
        row[label + '_mse'] = float(error[mask].mean())
        row[label + '_squared_error_sum'] = float(error[mask].sum())
        row[label + '_count'] = int(mask.sum())
    row['active_miss_rate'] = float((prediction[target > 0] < .5).mean())
    row['inactive_false_positive_rate'] = float((prediction[target == 0] > .5).mean())
    return row, error.mean(1)


def trained_models():
    result = {}
    weights = load_file(ROOT / 'reports/toy-scalar-comparison-v1/weights-100000.safetensors')
    aliases = {'relu': 'relu50', 'gelu': 'gelu50', 'silu': 'silu50',
               'leaky_relu': 'leakyrelu50', 'relu2': 'relu2_50'}
    for activation, group in aliases.items():
        for i, (loss, seed) in enumerate((loss, seed) for loss in [2, 4] for seed in range(3)):
            name = f'trained_{group}_L{loss}_s{seed}'
            model = {role: weights[f'{group}.{role}'][i] for role in ['gate', 'out']}
            result[name] = (model, {'track': 'bottleneck', 'activation': activation,
                                   'loss_exponent': loss, 'training_seed': seed, 'origin': 'trained'})
    for family, core in [('orthogonal', C), ('coupled', K)]:
        weights = load_file(core.OUT / 'weights-100000.safetensors')
        for activation in C.P['activations']:
            for i, (task, seed) in enumerate(C.members()):
                name = f'trained_{family}_{activation}_{task}_s{seed}'
                model = {role: weights[f'{activation}.{role}'][i] for role in ['gate', 'out', 'bias', 'out_bias']}
                result[name] = (model, {'track': 'binary', 'family': family, 'task': task,
                                       'activation': activation, 'training_seed': seed, 'origin': 'trained'})
    return result


def paired_summary(candidate, trained):
    # One observation is one independent latent/context/noise example. Averaging
    # three frozen trained models' errors does not pretend they are extra data.
    delta = candidate - np.mean(trained, axis=0)
    mean = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {'candidate_minus_mean_trained_mse': mean, 'evaluation_se': se,
            'normal_95_interval': [mean - 1.96*se, mean + 1.96*se], 'examples': len(delta)}


def write_report(out, rows, paired):
    def mean(selected, key): return float(np.mean([r[key] for r in selected]))
    lines = ['# Formula-written weights against trained toys', '',
             'Eleven actual weight sets, zero gradient updates. Three fresh evaluation seeds, '
             '16,384 examples each; raw outputs and the same inputs for every matched contestant.', '',
             'The written independent-code denoisers are competitive with trained ReLU models. '
             'The shared sparse code improves positive-feature accuracy but pays for it in inactive '
             'leakage and overall loss. Trained networks clearly outperform the written decoders '
             'when denoising overlapping features.', '',
             '![Fresh evaluation comparison](comparison.png)', '',
             '## 100 → 50 → 100 sparse ReLU task', '',
             'Positive MSE is the mean of per-feature positive-input MSEs, matching the existing evaluator. '
             'Coverage and error columns are means over evaluation seeds (and three training seeds for trained rows).', '',
             '| Model | Coverage / 100 | Positive MSE | Inactive RMSE | Total MSE | Fourth-power loss |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ['axis50', 'code5_gain09', 'code10_gain09', 'trained ReLU L2', 'trained ReLU L4']:
        selected = [r for r in rows if r['track'] == 'bottleneck' and
                    (r['name'] == name if not name.startswith('trained ') else
                     r['origin'] == 'trained' and r['activation'] == 'relu' and r['loss_exponent'] == int(name[-1]))]
        lines.append(f"| {name} | {mean(selected,'competent_features'):.1f} | {mean(selected,'positive_mse'):.6f} | "
                     f"{mean(selected,'inactive_rmse'):.6f} | {mean(selected,'all_mse'):.6f} | {mean(selected,'fourth_power_loss'):.8f} |")
    lines += ['', '## 33 → 64 → 16 binary decoders', '',
              'Each entry compares the constructed model with all three matching trained ReLU seeds. '
              'The exact posterior is a reference with unrestricted computation. Clean-trained models '
              'are shown on clean inputs; denoisers at their design noise σ=0.35. All models also get '
              'the full σ=0, 0.35, 0.6 sweep in metrics.json.', '',
              '| Family / task | Written MSE | Trained ReLU MSE | Exact posterior MSE | Written active MSE | Trained active MSE |',
              '|---|---:|---:|---:|---:|---:|']
    for family in ['orthogonal', 'coupled']:
        for task in C.P['tasks']:
            sigma = .35 if task.endswith('denoise') else 0
            matched = [r for r in rows if r.get('family') == family and r.get('task') == task and r.get('sigma') == sigma]
            manual = [r for r in matched if r['origin'] == 'written']
            trained = [r for r in matched if r['origin'] == 'trained' and r['activation'] == 'relu']
            oracle = [r for r in matched if r['origin'] == 'oracle']
            lines.append(f"| {family} / {task} | {mean(manual,'all_mse'):.7g} | {mean(trained,'all_mse'):.7g} | "
                         f"{mean(oracle,'all_mse'):.7g} | {mean(manual,'active_mse'):.6f} | {mean(trained,'active_mse'):.6f} |")
    lines += ['', '## How the weights are written', '',
              '- **axis50:** identity encoder/decoder on the first 50 coordinates; exactly discards the other 50.',
              '- **code5/code10:** fixed-seed positive binary columns with 5 or 10 entries per feature; '
              'a centered matrix pseudoinverse, scaled to isolated gain 0.9. This inverse acts on the '
              'constructed code, never on examples or trained weights.',
              '- **Clean binary decoder:** known encoder dual, followed by a two-ReLU clamp. Context '
              'branches use the context input directly in the hidden weights. Four neurons per feature '
              'suffice to route either context. All 65,536 clean bit patterns in both contexts are checked.',
              '- **Denoiser:** thresholds follow the known Bernoulli prior and Gaussian variance. '
              'Four ReLU hinges approximate the posterior sigmoid for simple tasks; two per context '
              'branch approximate it with a ramp. The overlap version uses marginal dual-coordinate '
              'variance and deliberately loses neighboring evidence.', '',
              '## Scope and evidence', '',
              'The constructor uses privileged task knowledge, especially the binary encoder. These '
              'are comparisons of resulting networks, not equally informed learning procedures. '
              'Both contestants use the same architecture, activation, output handling and evaluation samples.', '',
              'The sparse-code family was explored on a separate 8,192-example development set '
              '(ones=5/10/15/25, gain=0.9). Sizes 5 and 10 were retained before these fresh seeds were '
              'evaluated; this is exploratory selection. See the protocol and pilot.json for every pilot result. '
              'Binary formulas were fixed without a pilot. No fresh-test-based changes were made.', '',
              'All 30 bottleneck and 120 binary trained checkpoints are included in metrics.json. '
              'paired.json reports paired error differences against the mean error of the three matching '
              'trained ReLU models, with normal-approximation intervals over evaluation examples. '
              'These intervals do not describe training-seed or construction-seed uncertainty.', '',
              'The Gaussian context gate can fail in extreme unbounded tails; it is exact on clean '
              'binary codewords, not on all real inputs. The noise sweep includes σ=0.6 as a stress '
              'test. Feature-boundary p is not measured in this task-performance experiment.', '',
              'Weights: weights.safetensors. Construction: scripts/write_toy_weights.py. '
              'Evaluation: scripts/evaluate_written_toys.py. Checks: tests/test_written_toys.py. '
              'manifest.json records input, source, protocol and output hashes.', '',
              'Reproduce into a new directory (Python 3.12, requirements/toy.txt):', '',
              '```bash', 'python scripts/evaluate_written_toys.py --out work/handwritten-reproduction',
              'python -m unittest discover -s tests -p test_written_toys.py', '```', '']
    (out / 'findings.md').write_text('\n'.join(lines))


@torch.inference_mode()
def main(out):
    torch.set_num_threads(P['evaluation']['cpu_threads'])
    # Serialize and seal written tensors BEFORE loading baselines or generating fresh data.
    _, metadata = save_models(out)
    arrays = load_file(out / 'weights.safetensors')
    models = {name: ({role.split('.')[-1]: value for role, value in arrays.items()
                      if role.startswith(name + '.')}, {**meta, 'origin': 'written'})
              for name, meta in metadata.items()}
    source_paths = [PROTOCOL, Path(__file__).resolve(), ROOT/'scripts/write_toy_weights.py',
                    ROOT/'scripts/characterization_core.py', ROOT/'scripts/coupled_characterization.py',
                    ROOT/'scripts/plot_written_toys.py', ROOT/'tests/test_written_toys.py']
    checkpoints = [ROOT/'reports/toy-scalar-comparison-v1/weights-100000.safetensors',
                   C.OUT/'weights-100000.safetensors', K.OUT/'weights-100000.safetensors']
    seal = {'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'sources': {str(p.relative_to(ROOT)): C.sha(p) for p in source_paths},
            'written_weights_sha256': C.sha(out/'weights.safetensors'),
            'baseline_weights': {str(p.relative_to(ROOT)): C.sha(p) for p in checkpoints},
            'protocol': P, 'numpy': np.__version__, 'torch': torch.__version__}
    C.write(out/'input-seal.json', seal)
    C.write(out/'pilot.json', P['pilot'])
    models.update(trained_models())
    rows = []; paired = []
    for seed in P['evaluation']['seeds']:
        rng = np.random.default_rng(seed); n = P['evaluation']['samples_per_seed']
        x = ((rng.random((n, 100)) < .02) * rng.uniform(-1, 1, (n, 100))).astype(np.float32)
        for name, (model, meta) in models.items():
            if meta['track'] != 'bottleneck': continue
            prediction = predict(x, model, meta['activation'])
            rows.append({'name': name, **meta, 'eval_seed': seed,
                         **bottleneck_metrics(x, prediction, model, meta['activation'])})
        bits = (rng.random((n, 16)) < .125).astype(np.float32)
        context = rng.choice([-1., 1.], n).astype(np.float32)
        noise = rng.normal(size=(n, 32)).astype(np.float32)
        for family, core in [('orthogonal', C), ('coupled', K)]:
            a = core.encoder()
            for task in C.P['tasks']:
                base = C.task_inputs(torch.from_numpy(bits), torch.from_numpy(context),
                                    torch.tensor(a, dtype=torch.float32), task).numpy()
                for sigma in P['evaluation']['binary_noise_std']:
                    x = base.copy(); x[:, :32] += sigma * noise
                    errors = {}
                    for name, (model, meta) in models.items():
                        if meta.get('family') != family or meta.get('task') != task: continue
                        prediction = predict(x, model, meta['activation'])
                        metrics, loss = binary_metrics(bits, prediction)
                        if meta['activation'] == 'relu': errors[name] = loss
                        rows.append({'name': name, **meta, 'eval_seed': seed, 'sigma': sigma, **metrics})
                    _, posterior = core.posterior_mean(x[:, :32].astype(float), a, context, task, sigma)
                    metrics, _ = binary_metrics(bits, posterior)
                    rows.append({'name': 'exact_posterior', 'track': 'binary', 'origin': 'oracle',
                                 'family': family, 'task': task, 'sigma': sigma, 'eval_seed': seed, **metrics})
                    paired.append({'family': family, 'task': task, 'sigma': sigma, 'eval_seed': seed,
                                   **paired_summary(errors[f'{family}_{task}'],
                                       [errors[f'trained_{family}_relu_{task}_s{s}'] for s in range(3)])})
        print(f'Evaluated fresh seed {seed}: {len(rows)} records', flush=True)
    C.write(out/'metrics.json', rows); C.write(out/'paired.json', paired)
    scalar_keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, list)})
    with (out/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys); writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k in scalar_keys} for row in rows)
    write_report(out, rows, paired)
    from plot_written_toys import plot
    plot(out)
    C.write(out/'manifest.json', {'input_seal_sha256': C.sha(out/'input-seal.json'),
                                'files': {p.name: C.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}})
    print(f'Complete: {out}/findings.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory, never overwritten.')
    args = parser.parse_args()
    main(args.out.resolve())
