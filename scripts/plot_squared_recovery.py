"""Plot the tiny matched-control/recovery assay from saved data (no torch)."""
from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'trained': '#9364a4', 'independent': '#c47d35', 'input_reference': '#555555'}
LABELS = {'trained': 'Trained squared ReLU', 'independent': 'Independent control',
          'input_reference': 'Task on noisy input'}


def main(out):
    geometry = json.loads((out / 'geometry.json').read_text())['records']
    corruption = json.loads((out / 'corruption.json').read_text())['records']
    protocol = json.loads((ROOT / 'configs/squared-recovery-v1.json').read_text())
    prepared = json.loads((out / 'prepared.json').read_text())
    spec = importlib.util.spec_from_file_location('fsec_fit', ROOT / 'vendor/fsec/scripts/lib/superellipse.py')
    fitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitter)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    with np.load(out / 'geometry.npz') as raw:
        for name in ('trained', 'independent'):
            index = next(i for i, row in enumerate(geometry) if row['model'] == name
                         and row['kind'] == 'encoder_columns' and row['anchor_value'] == 0)
            fit = geometry[index]['fits']['0.5']
            points = fitter.extract_contour(raw['axis'], np.median(raw['l2'][index], axis=0), fit['threshold'])
            points /= np.array(fit['intercepts_feature_units'])
            points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
            axes[0].plot(points[:, 0], points[:, 1], color=COLORS[name],
                         ls='-' if name == 'trained' else '--', lw=2,
                         marker='o' if name == 'trained' else None, markevery=6, ms=3,
                         label=f'{LABELS[name]}: p={fit["p"]:.3f}')
    angle = np.linspace(0, np.pi / 2, 100)
    axes[0].plot(np.cos(angle), np.sin(angle), color='#aaaaaa', lw=1, ls=':', label='Round reference, p=2')
    axes[0].set(title='Same squarer boundary', xlabel='Feature A / axis cutoff',
                ylabel='Feature B / axis cutoff', xlim=(0, 1.08), ylim=(0, 1.08), aspect='equal')
    axes[0].legend(fontsize=7, loc='lower left')
    for ax, metric, title in zip(axes[1:], ['all_mse', 'active_mse'],
                                ['All 50 supported outputs', 'Only real positive signals']):
        for name in ('trained', 'independent', 'input_reference'):
            rows = [row for row in corruption if row['model'] == name and
                    row['method'] in ('natural_squared_relu', 'task_on_corrupted_input')]
            rows.sort(key=lambda row: row['sigma'])
            ax.plot([row['sigma'] for row in rows], [row[metric] for row in rows],
                    color=COLORS[name], label=LABELS[name], marker='o', ms=4,
                    ls='--' if name == 'independent' else '-', lw=1.8)
        ax.set(title=title, xlabel='Input noise per coordinate (std)', ylabel='Error to fixed clean target (MSE)')
        ax.legend(fontsize=7)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15)
    fig.suptitle('Boundary shape and actual corruption recovery are separate tests', fontsize=13)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.text(.5, .025, 'One specialized checkpoint; one no-cross-feature control. Lower error is better.\n'
             'Clean target stays fixed; average error includes many absent features. These models were not trained to denoise.',
             ha='center', fontsize=9, color='#555555')
    fig.savefig(out / 'comparison.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    anchors = protocol['geometry']['anchor_values']
    for name in ('trained', 'independent'):
        fits = [next(row['fits']['0.5'] for row in geometry if row['model'] == name
                     and row['kind'] == 'encoder_columns' and row['anchor_value'] == value) for value in anchors]
        ax.plot(range(3), [row['p'] for row in fits], color=COLORS[name], marker='o',
                ls='--' if name == 'independent' else '-', label=LABELS[name])
    ax.set(xticks=range(3), xticklabels=['Absent\n0', 'Positive\n+0.5', 'Negative\n−0.5'],
           xlabel='Starting values of the same two features', ylabel='Fitted boundary exponent p',
           title='The operating point moves p in both models')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'operating-point.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    rows = []
    for record in corruption:
        clean = next(row for row in corruption if row['model'] == record['model'] and
                     row['method'] == record['method'] and row['sigma'] == 0)
        rows.append({**record, **{metric + '_increase_from_clean': record[metric] - clean[metric]
                     for metric in ['all_mse', 'active_mse', 'absent_mse', 'full_mse']}})
    (out / 'noise-penalties.json').write_text(json.dumps(rows, indent=2) + '\n')
    primary = protocol['corruption']['primary_std']
    selected = [row for row in rows if row['sigma'] == primary and
                row['method'] in ('natural_squared_relu', 'task_on_corrupted_input')]
    lines = ['# Primary fixed-target corruption scores', '',
             f'Input noise std {primary}; {protocol["corruption"]["samples"]} clean examples × 4 paired draws.', '',
             '| Model | All supported MSE | Positive-active MSE | Absent MSE | Positive-active increase from clean |',
             '|---|---:|---:|---:|---:|']
    for row in selected:
        lines.append('| ' + LABELS[row['model']] + ' | ' + ' | '.join(f'{row[key]:.7f}' for key in
                     ['all_mse', 'active_mse', 'absent_mse', 'active_mse_increase_from_clean']) + ' |')
    lines += ['', 'All supported = the same 50 competent output features. Full 100-output errors, oracle ablations,',
              'zero-noise errors, all noise levels, flags and paired bootstrap differences remain in the saved JSON.', '']
    (out / 'numerical-summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/squared-recovery-v1')
    main(parser.parse_args().out.resolve())
