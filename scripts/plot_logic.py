"""CPU-only saved-data summaries and three figures for the AND/XOR pilot."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics as S

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'relu': '#626262', 'gelu': '#377bad', 'silu': '#368866', 'leaky_relu': '#b66d3e', 'relu2': '#9364a4'}
LABELS = {'relu': 'ReLU', 'gelu': 'GELU', 'silu': 'SiLU', 'leaky_relu': 'Leaky ReLU', 'relu2': 'Squared ReLU'}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15, axis='y')


def get_score(record, which='natural', sigma=.35):
    return next(row for row in record['recovery'] if row['map'] == which and row['sigma'] == sigma)


def summarize(source):
    records = [json.loads(path.read_text()) for path in sorted((source / 'analysis').glob('*_t*.json'))]
    controls = json.loads((source / 'projection-check.json').read_text())['records']
    assert len(records) == 180
    networks = []
    for record in records:
        meta = record['network']; row = dict(meta)
        for which in ['natural', 'clipped_natural', 'plugin_reference', 'Bayes_oracle', 'constant_prior']:
            for metric in ['all_mse', 'positive_mse', 'negative_mse', 'balanced_mse', 'positive_miss_rate', 'negative_false_positive_rate']:
                row[which + '_' + metric] = get_score(record, which)[metric]
        row['clean_mse'] = get_score(record, 'natural', 0.)['all_mse']
        row['clean_balanced_mse'] = get_score(record, 'natural', 0.)['balanced_mse']
        row['noise_penalty'] = row['natural_all_mse'] - row['clean_mse']
        for metric in ['all_mse', 'positive_mse', 'negative_mse', 'balanced_mse']:
            row['projected_clipped_' + metric] = None
            if meta['step'] == 30000:
                score = next(r for r in controls if r['network']['key'] == meta['key'] and r['sigma'] == .35
                             and r['map'] == 'projected_clipped_natural')
                row['projected_clipped_' + metric] = score[metric]
        for name in ['within', 'across']:
            case = next(case for case in record['geometry'] if case['pair']['name'] == name and case['state'] == [0, 0])
            selected = case['backgrounds'][3]; fit = selected['fit']
            row[name + '_p'] = fit['p']; row[name + '_flags'] = fit['all_flags']
            row[name + '_interaction'] = selected['mean_vector_interaction_residual']
            row[name + '_axis_null_p'] = selected['axis_vector_null_fit']['p']
            if name == 'within': row['within_output0_flags'] = selected['output0_only_fit']['all_flags']
        networks.append(row)
    groups = []
    for task in ['and_clean', 'and_denoise', 'xor_clean', 'xor_denoise']:
        for name in COLORS:
            for step in [0, 5000, 30000]:
                selected = [r for r in networks if r['task'] == task and r['activation'] == name and r['step'] == step]
                row = {'task': task, 'activation': name, 'step': step, 'seeds': len(selected)}
                for key in ['natural_all_mse', 'clipped_natural_all_mse', 'natural_positive_mse', 'natural_negative_mse',
                            'natural_balanced_mse', 'natural_positive_miss_rate', 'natural_negative_false_positive_rate',
                            'Bayes_oracle_all_mse', 'plugin_reference_all_mse', 'constant_prior_all_mse', 'clean_mse']:
                    values = [r[key] for r in selected]
                    row[key] = {'mean': S.mean(values), 'min': min(values), 'max': max(values)}
                for pair in ['within', 'across']:
                    good = [r[pair + '_p'] for r in selected if r[pair + '_p'] is not None and not r[pair + '_flags']]
                    row[pair + '_p_good'] = {'count': len(good), 'mean': S.mean(good) if good else None}
                groups.append(row)
    return networks, groups


def plot(source, out):
    if out != source:
        assert out.is_relative_to((ROOT / 'work').resolve()), '--out must be under work/'
        assert not out.exists(), 'Choose a fresh --out directory.'
        out.mkdir(parents=True)
    networks, groups = summarize(source)
    (out / 'networks.json').write_text(json.dumps(networks, indent=2) + '\n')
    (out / 'seed-summary.json').write_text(json.dumps(groups, indent=2) + '\n')
    with (out / 'networks.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(networks[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in networks)
    final = [r for r in networks if r['step'] == 30000]
    lines = ['# Numerical summary', '', 'Final checkpoints; means over five activations and three initialization seeds. Noise std 0.35. MSE targets remain clean logical labels.', '',
             '| Task / training | Raw MSE | Clipped MSE | Projected + clipped MSE | Valid within fits | Valid across fits |',
             '|---|---:|---:|---:|---:|---:|']
    for task in ['and_clean', 'and_denoise', 'xor_clean', 'xor_denoise']:
        selected = [r for r in final if r['task'] == task]
        values = [S.mean(r[key] for r in selected) for key in ['natural_all_mse', 'clipped_natural_all_mse', 'projected_clipped_all_mse']]
        good = [sum(r[pair + '_p'] is not None and not r[pair + '_flags'] for r in selected) for pair in ['within', 'across']]
        lines.append(f'| {task} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {good[0]}/15 | {good[1]}/15 |')
    lines += ['', 'Noise-trained activations; each cell uses three initialization seeds. Geometry uses the across pair at the prespecified 00/background-3 slice. Only unflagged p values contribute to p means; validity counts are shown.', '',
              '| Activation | AND p (valid) | AND projected/clipped MSE | XOR p (valid) | XOR projected/clipped MSE |',
              '|---|---:|---:|---:|---:|']
    for name in COLORS:
        values = []
        for task in ['and_denoise', 'xor_denoise']:
            selected = [r for r in final if r['activation'] == name and r['task'] == task]
            ps = [r['across_p'] for r in selected if r['across_p'] is not None and not r['across_flags']]
            values += [(f'{S.mean(ps):.3f} ({len(ps)}/3)' if ps else 'NA (0/3)'), f'{S.mean(r["projected_clipped_all_mse"] for r in selected):.6f}']
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'Projected/clipped class errors:', '', '| Task / training | Positive MSE | Negative MSE | Class-balanced MSE |', '|---|---:|---:|---:|']
    for task in ['and_clean', 'and_denoise', 'xor_clean', 'xor_denoise']:
        selected = [r for r in final if r['task'] == task]
        values = [S.mean(r['projected_clipped_' + key] for r in selected) for key in ['positive_mse', 'negative_mse', 'balanced_mse']]
        lines.append('| ' + task + ' | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    (out / 'numerical-summary.md').write_text('\n'.join(lines) + '\n')
    p_top = max(6., 1.1 * max(r[pair + '_p'] for r in final for pair in ['within', 'across'] if r[pair + '_p'] is not None))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7))
    for ri, task in enumerate(['and', 'xor']):
        for offset, regime, alpha in [(-.18, 'clean', .4), (.18, 'denoise', 1.)]:
            selected = [r for r in final if r['task'] == task + '_' + regime]
            means = [S.mean(r['projected_clipped_all_mse'] for r in selected if r['activation'] == name) for name in COLORS]
            low = [mean - min(r['projected_clipped_all_mse'] for r in selected if r['activation'] == name) for name, mean in zip(COLORS, means)]
            high = [max(r['projected_clipped_all_mse'] for r in selected if r['activation'] == name) - mean for name, mean in zip(COLORS, means)]
            axes[ri, 0].bar(np.arange(5) + offset, means, .33, color=list(COLORS.values()), alpha=alpha,
                           yerr=[low, high], capsize=2, label='Clean training' if regime == 'clean' else 'Noise training')
            for ci, pair in enumerate(['within', 'across'], 1):
                for ai, name in enumerate(COLORS):
                    for r in selected:
                        if r['activation'] != name or r[pair + '_p'] is None: continue
                        good = not r[pair + '_flags']
                        axes[ri, ci].scatter(ai + offset + .055 * (r['seed'] - 1), r[pair + '_p'], s=28,
                                            marker='o' if regime == 'clean' else '^',
                                            facecolor=COLORS[name] if good else 'none', edgecolor=COLORS[name], alpha=.9)
        example = next(r for r in final if r['task'].startswith(task))
        axes[ri, 0].axhline(example['Bayes_oracle_all_mse'], color='#368866', ls='--', label='Exact denoiser')
        axes[ri, 0].axhline(example['plugin_reference_all_mse'], color='#888888', ls=':', label='Linear-decode logic reference')
        axes[ri, 0].set(ylabel=task.upper() + '\nFixed-target noisy MSE', title='Recovery after projection + clipping')
        for ci, pair in enumerate(['within', 'across'], 1):
            axes[ri, ci].axhline(2, color='#aaaaaa', ls=':', lw=1)
            axes[ri, ci].set(title='Two bits feeding one output' if pair == 'within' else 'Bits feeding separate outputs',
                             ylabel=task.upper() + '\nNominal fitted p', ylim=(.4, p_top))
        for ax in axes[ri]:
            ax.set_xticks(range(5), [LABELS[name].replace(' ', '\n') for name in COLORS], fontsize=8); style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle('AND/XOR: task recovery and conditional geometry\nRecovery: noise std 0.35, mean/range of 3 seeds. Geometry: circles clean-trained, triangles noise-trained; hollow = flagged.', fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .91))
    fig.text(.5, .055, 'Recovery panels use the final-only fairness check. Geometry uses the primary 00 anchor/background 3; across-pair AND partners are on.\nFlagged p values are not reliable exponents; missing points have no fitted exponent.', ha='center', fontsize=9)
    fig.savefig(out / 'overview.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    baselines = json.loads((source / 'analysis/baselines.json').read_text())
    with np.load(source / 'analysis/baselines.npz') as reference:
        for ri, task in enumerate(['and', 'xor']):
            key = f'relu_{task}_denoise_s0_t030000'
            record = json.loads((source / 'analysis' / (key + '.json')).read_text())
            oracle = next(r for r in baselines['records'] if r['task'] == task and r['map'] == 'Bayes_oracle')
            with np.load(source / 'analysis' / (key + '.npz')) as raw:
                for ci, pair in enumerate(['within', 'across']):
                    index = next(i for i, r in enumerate(record['geometry']) if r['pair']['name'] == pair and r['state'] == [0, 0])
                    case = record['geometry'][index]; fit = case['backgrounds'][3]['fit']
                    oracle_index = next(i for i, r in enumerate(oracle['geometry']) if r['pair']['name'] == pair and r['state'] == [0, 0])
                    oracle_fit = oracle['geometry'][oracle_index]['backgrounds'][3]['fit']
                    grid = raw['response'][index, 3]; ref = reference[task + '_Bayes_oracle_response'][oracle_index, 3]
                    ax = axes[ri, ci]; axis = np.linspace(0, 1, 31)
                    ax.pcolormesh(axis, axis, (grid / max(grid.max(), 1e-15)).T, shading='auto', cmap='viridis', vmin=0, vmax=1)
                    for values, result, color, line in [(grid, fit, 'white', '-'), (ref, oracle_fit, '#ffb14e', '--')]:
                        if result['threshold'] > 1e-10:
                            ax.contour(axis, axis, values.T, levels=[result['threshold']], colors=[color], linestyles=line, linewidths=2)
                    measured = 'fit flagged' if fit['all_flags'] else f'p={fit["p"]:.2f}'
                    exact = 'fit flagged' if oracle_fit['all_flags'] else f'p={oracle_fit["p"]:.2f}'
                    ax.set(title=f'{task.upper()}, {pair} pair\nModel: {measured}; exact denoiser: {exact}',
                           xlabel='First bit (0 → 1)', ylabel='Second bit (0 → 1)', aspect='equal')
    fig.suptitle('ReLU, noise-trained, seed 0: same assay on a network and an exact denoiser', fontsize=12)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.text(.5, .035, 'Heat: model output-change norm / grid maximum. White contour: model; orange dashed: exact denoiser.\nEach contour uses half its own smaller maximum axis response. Interior curves are continuous probes between Boolean corners.', ha='center', fontsize=9)
    fig.savefig(out / 'contours.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey='row')
    for ri, task in enumerate(['and', 'xor']):
        for ci, regime in enumerate(['clean', 'denoise']):
            ax = axes[ri, ci]
            for name, color in COLORS.items():
                selected = [r for r in networks if r['task'] == task + '_' + regime and r['activation'] == name]
                steps = [0, 5000, 30000]
                for seed in [0, 1, 2]:
                    values = [next(r['natural_all_mse'] for r in selected if r['seed'] == seed and r['step'] == step) for step in steps]
                    ax.plot(np.array(steps) + 1, values, color=color, alpha=.2, lw=.7)
                ax.plot(np.array(steps) + 1, [S.mean(r['natural_all_mse'] for r in selected if r['step'] == step) for step in steps], color=color, marker='o', label=LABELS[name])
            ref = next(r for r in networks if r['task'] == task + '_' + regime)
            ax.axhline(ref['Bayes_oracle_all_mse'], color='#368866', ls='--', lw=1)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1, 5001, 30001], ['0', '5k', '30k'])
            ax.set(title=task.upper() + ', ' + ('clean training' if regime == 'clean' else 'noise training'),
                   xlabel='Training updates', ylabel='Unclipped noisy MSE (std 0.35)'); style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    fig.suptitle('Learning the clean truth table and recovering corrupted answers\nThick: seed mean; faint: individual seeds; dashed green: exact denoiser.', fontsize=12)
    fig.tight_layout(rect=(0, .055, 1, .91)); fig.savefig(out / 'training-recovery.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    receipt = {'source': str(source.relative_to(ROOT)), 'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'networks': 60, 'frames': 180, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
               'figures': ['overview.png', 'contours.png', 'training-recovery.png'], 'meaning': 'Saved-data summaries and redraw; no training or fit changes.'}
    (out / 'plot-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('PLOTS_COMPLETE', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/toy-logic-v1')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(); source = args.source.resolve(); plot(source, args.out.resolve() if args.out else source)
