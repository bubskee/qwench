"""Plot the completed toy characterization from saved JSON/NPZ, without torch."""
from pathlib import Path
import importlib.util
import json
import statistics as S
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('QWENCH_STUDY_OUT', str(ROOT / 'reports/toy-characterization-study-v1'))).resolve()
FOLDERS = {'orthogonal': 'toy-characterization-v1', 'coupled': 'toy-characterization-coupled-v1'}
COLORS = {'relu': '#666666', 'gelu': '#377bad', 'silu': '#368866', 'leaky_relu': '#b66d3e', 'relu2': '#9364a4'}
LABELS = {'relu': 'ReLU', 'gelu': 'GELU', 'silu': 'SiLU', 'leaky_relu': 'Leaky ReLU', 'relu2': 'Squared ReLU'}
TASKS = ['linear_clean', 'linear_denoise', 'context_clean', 'context_denoise']
TITLES = ['Simple decoder / clean training', 'Simple decoder / noise training', 'Context decoder / clean training', 'Context decoder / noise training']


def style(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15)


def main():
    records = json.loads((OUT / 'networks.json').read_text())
    spec = importlib.util.spec_from_file_location('fsec', ROOT / 'vendor/fsec/scripts/lib/superellipse.py')
    fitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitter)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, family in enumerate(FOLDERS):
        source = ROOT / 'reports' / FOLDERS[family]
        ident = 'new_relu_linear_denoise_s0_t100000'
        data = json.loads((source / 'analysis/new' / (ident + '.json')).read_text())
        with np.load(source / 'analysis/new' / (ident + '.npz')) as raw:
            for which, color, line, label in [('natural', '#9364a4', '-', 'Trained model'), ('separable', '#c47d35', '--', 'Separable control')]:
                i = next(i for i, row in enumerate(data['geometry']) if row['pair'] == [4, 9] and row['anchor'] == 0
                         and row['sign'] == [1, 1] and row['kind'] == 'feature_axes' and row['map'] == which)
                fit = data['geometry'][i]['fits']['0.5']
                points = fitter.extract_contour(np.linspace(0, 1.5, 31), np.median(raw['response'][i], axis=0), fit['threshold'])
                points /= np.array(fit['intercepts_feature_units']); points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
                axes[0, col].plot(points[:, 0], points[:, 1], color=color, ls=line, lw=2, label=f'{label}: p={fit["p"]:.2f}')
        ax = axes[0, col]; ax.set(title='Independent features' if col == 0 else 'Overlapping feature code',
            xlabel='Feature A / axis cutoff', ylabel='Feature B / axis cutoff', aspect='equal', xlim=(0, 1.07), ylim=(0, 1.07)); ax.legend(fontsize=8); style(ax)
        selected = [r for r in records if r['family'] == family and r['task'] == 'linear_denoise' and r['activation'] == 'relu' and r['step'] == 100000]
        keys = ['natural_all_mse', 'separable_all_mse', 'Bayes_mse', 'reference_mse']
        means = [S.mean(r[k] for r in selected) for k in keys]
        low = [m - min(r[k] for r in selected) for k, m in zip(keys, means)]
        high = [max(r[k] for r in selected) - m for k, m in zip(keys, means)]
        ax = axes[1, col]; ax.bar(range(4), means, color=['#9364a4', '#c47d35', '#368866', '#aaaaaa'], yerr=[low, high], capsize=3)
        ax.set(xticks=range(4), xticklabels=['Trained', 'Separable\ncontrol', 'Exact\ndenoiser', 'Linear\nreference'],
               ylabel='Fixed-target MSE (lower is better)', title='Actual corruption recovery, noise std 0.35', ylim=(0, .105)); style(ax)
    fig.suptitle('Similar squarer geometry; different benefit from cross-feature information', fontsize=14)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.text(.5, .02, 'ReLU example. Contours: seed 0, same low-overlap pair (4,9), absent ++ slice.\nBars: mean/range over 3 initialization seeds. The control preserves the trained isolated-feature response curves.', ha='center', fontsize=9)
    fig.savefig(OUT / 'matched-example.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    for metric, filename, heading in [('natural_p', 'training-geometry.png', 'Geometry through training'),
                                      ('natural_all_mse', 'training-recovery.png', 'Held-out corruption recovery through training')]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
        for ri, family in enumerate(FOLDERS):
            for ci, task in enumerate(TASKS):
                ax = axes[ri, ci]
                for name, color in COLORS.items():
                    selected = [r for r in records if r['family'] == family and r['task'] == task and r['activation'] == name]
                    steps = sorted(set(r['step'] for r in selected)); means = []
                    for seed in [0, 1, 2]:
                        values = [next(r[metric] for r in selected if r['seed'] == seed and r['step'] == step) for step in steps]
                        ax.plot(np.array(steps) + 1, values, color=color, alpha=.16, lw=.7)
                    for step in steps:
                        values = [r[metric] for r in selected if r['step'] == step and r[metric] is not None]
                        means.append(S.mean(values) if values else np.nan)
                    ax.plot(np.array(steps) + 1, means, color=color, lw=1.6, label=LABELS[name])
                    if metric == 'natural_p':
                        for step, value in zip(steps, means):
                            flagged = any(r['natural_flagged_fits'] for r in selected if r['step'] == step)
                            ax.scatter(step + 1, value, s=20, facecolor='none' if flagged else color, edgecolor=color, zorder=3)
                if metric != 'natural_p':
                    example = next(r for r in records if r['family'] == family and r['task'] == task)
                    ax.axhline(example['reference_mse'], color='#444444', ls=':', lw=1, label='Clipped linear reference')
                    ax.axhline(example['Bayes_mse'], color='#368866', ls='--', lw=1, label='Exact Bayes denoiser')
                    ax.set_yscale('log')
                else: ax.set_ylim(.4, 10.1); ax.axhline(2, color='#aaaaaa', ls=':', lw=1)
                ax.set_xscale('log'); ax.set_xticks([1, 101, 1001, 5001, 20001, 100001], ['0', '100', '1k', '5k', '20k', '100k'], rotation=35)
                ax.set(title=TITLES[ci] if ri == 0 else '', xlabel='Training updates',
                       ylabel=('Independent code' if ri == 0 else 'Overlapping code') + '\n' + ('Nominal fitted p' if metric == 'natural_p' else 'Noisy MSE (std 0.35)')); style(ax)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), fontsize=8)
        subtitle = 'Hollow points include flagged fits; all nominal fits retained. Thick: seed mean; faint: individual seeds.' if metric == 'natural_p' else 'Clean target stays fixed. Thick: mean of 3 initialization seeds; faint: individual seeds. Lower is better.'
        fig.suptitle(heading + '\n' + subtitle, fontsize=13)
        fig.tight_layout(rect=(0, .055, 1, .93)); fig.savefig(OUT / filename, dpi=160, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for family in FOLDERS:
        color = '#377bad' if family == 'orthogonal' else '#9364a4'
        null_data = json.loads((ROOT / 'reports/toy-characterization-axis-null-v1' / (family + '.json')).read_text())
        for r in null_data['records']:
            if r['network']['track'] != 'new' or r['network']['step'] != 100000 or r['map'] != 'natural': continue
            p = r['original_fits']['0.5']; q = r['axis_null_fits']['0.5']
            if p['p'] is None or q['p'] is None: continue
            good = not p['flags'] and not q['flags']
            axes[0].scatter(p['p'], q['p'], s=22, facecolors=color if good else 'none', edgecolors=color, alpha=.65)
        for r in records:
            if r['family'] != family or r['step'] != 100000 or not r['task'].endswith('denoise') or r['p_minus_separable'] is None: continue
            good = r['natural_good_fits'] == 2 and r['separable_good_fits'] == 2
            axes[1].scatter(r['p_minus_separable'], r['clean_adjusted_advantage'], s=30,
                            marker='D' if r['task'].startswith('context') else 'o', facecolors=color if good else 'none', edgecolors=color, alpha=.8)
    axes[0].plot([1, 8], [1, 8], color='#aaaaaa', ls='--'); axes[0].set(title='One-axis curves explain much of p', xlabel='Measured p', ylabel='p predicted from single-axis norms')
    axes[1].axhline(0, color='#aaaaaa', ls='--'); axes[1].axvline(0, color='#aaaaaa', ls=':')
    axes[1].set(title='Additional recovery need not increase p', xlabel='p(model) − p(separable control)', ylabel='Control noise penalty − model noise penalty')
    for ax in axes: style(ax)
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#377bad',label='Independent code'), Line2D([0],[0],marker='o',ls='',color='#9364a4',label='Overlapping code')], loc='lower center', ncol=2)
    fig.suptitle('All five activations and three seeds; final checkpoint\nHollow markers include flagged fits. Right: circles simple, diamonds context-dependent.', fontsize=12)
    fig.tight_layout(rect=(0, .075, 1, .89)); fig.savefig(OUT / 'geometry-explanation.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ci, task in enumerate(['L2', 'L4']):
        for ri, metric in enumerate(['natural_p', 'natural_active_mse']):
            ax = axes[ri, ci]
            for name, color in COLORS.items():
                subset = [r for r in records if r['family'] == 'historical' and r['activation'] == name and r['task'] == task]
                steps = sorted(set(r['step'] for r in subset))
                values = [S.mean(r[metric] for r in subset if r['step'] == step and r[metric] is not None) for step in steps]
                ax.plot(steps, values, color=color, marker='o', label=LABELS[name])
            ax.set(title=f'Historical {task}' if ri == 0 else '', xlabel='Training updates', ylabel='Nominal p' if ri == 0 else 'Positive-active noisy MSE'); style(ax)
    fig.suptitle('Original signed-sparse task: late saved checkpoints\nTwo fixed pairs, all fits retained; no competence-based promotion. Noise std 0.075.', fontsize=12)
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .92)); fig.savefig(OUT / 'historical-trajectories.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, family in zip(axes, FOLDERS):
        data = json.loads((ROOT / 'reports/toy-characterization-pair-diagnostic-v1' / (family + '.json')).read_text())
        for offset, pair, marker, line in [(-.08, [0,1], 'o', '-'), (.08, [4,9], 'D', '--')]:
            means=[]; lows=[]; highs=[]
            for name in COLORS:
                values = [r['mean_vector_interaction_residual'] for r in data['records'] if r['network']['activation'] == name
                          and r['network']['task'] == 'linear_denoise' and r['pair'] == pair and r['sign'] == [1,1] and r['map'] == 'natural']
                mean = S.mean(values); means.append(mean); lows.append(mean-min(values)); highs.append(max(values)-mean)
            ax.errorbar(np.arange(5)+offset, means, yerr=[lows,highs], marker=marker, ls=line, capsize=3,
                        label='Pair (0,1)' + ('; directly linked' if family=='coupled' else '') if pair == [0,1] else 'Pair (4,9); separate blocks')
        ax.set(title='Independent code' if family=='orthogonal' else 'Overlapping code', xticks=range(5),
               xticklabels=[LABELS[n].replace(' ', '\n') for n in COLORS], ylabel='Normalized nonlinear interaction residual')
        ax.legend(fontsize=8); style(ax)
    fig.suptitle('Match full single-axis output vectors, then test their combination', fontsize=13)
    fig.tight_layout(rect=(0, .10, 1, .93))
    fig.text(.5, .025, 'Noise-trained simple decoder, mean/range of 3 seeds. Residual zero means exact vector additivity.\nThe directly linked pair has input cosine 0.6 in the overlap code: outside the paper’s low-overlap filter.', ha='center', fontsize=9)
    fig.savefig(OUT / 'pair-interaction.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    print('PLOTS_COMPLETE: six characterization figures')


if __name__ == '__main__': main()
