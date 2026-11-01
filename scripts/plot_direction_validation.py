"""Plot observed direction checks; no model inference or fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/directions'
versions = [('v1', 'Block 2'), ('v2', 'Block 8')]
colors = {'sentiment': '#b75a45', 'language': '#367a99'}
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), gridspec_kw={'width_ratios': [1, .8, 1.6]})
for version_index, (version, label) in enumerate(versions):
    folder = OUT / version
    extraction = json.loads((folder / 'extraction.json').read_text())
    causal = json.loads((folder / 'causal-summary.json').read_text())
    for target_index, target in enumerate(['sentiment', 'language']):
        x = version_index * 3 + target_index
        accuracy = extraction['metrics'][target]['heldout_accuracy']
        axes[0].bar(x, accuracy, color=colors[target], width=.72)
        axes[0].text(x, accuracy + .025, f'{accuracy:.0%}', ha='center', fontsize=10)
        values = causal[target]['directions'][target]['values']
        jitter = np.linspace(-.17, .17, len(values))
        axes[2].scatter(x + jitter, values, color=colors[target], s=25, zorder=3)
        axes[2].plot([x - .27, x + .27], [np.median(values)] * 2, color=colors[target], linewidth=2.4)
        for control_index, control in enumerate(['random_101', 'random_202']):
            random_values = causal[target]['directions'][control]['values']
            axes[2].scatter(x + jitter + .025 * (control_index - .5), random_values,
                            color='#a3a3a3', marker='x', s=15, alpha=.65, zorder=2)
    overlap = abs(extraction['raw_cosine'])
    axes[1].bar(version_index, overlap, color='#706b83', width=.6)
    axes[1].text(version_index, overlap + .006, f'{overlap:.3f}', ha='center', fontsize=10)

labels = ['Sent.\nBlock 2', 'Lang.\nBlock 2', 'Sent.\nBlock 8', 'Lang.\nBlock 8']
positions = [0, 1, 3, 4]
axes[0].set(xticks=positions, xticklabels=labels, ylim=(0, 1.13), ylabel='Held-out classification accuracy', title='Can the arrow recognize it?')
axes[0].axhline(.5, color='#777777', linestyle=':', label='Chance')
axes[0].axhline(.75, color='#777777', linestyle='--', label='Preset minimum')
axes[0].legend(loc='lower left', fontsize=8)
axes[1].set(xticks=[0, 1], xticklabels=['Block 2', 'Block 8'], ylabel='Absolute cosine overlap', title='Are the arrows distinct?')
axes[1].axhline(.1, color='#b75a45', linestyle='--', label='Maximum allowed')
axes[1].set_ylim(0, max(.22, axes[1].get_ylim()[1] * 1.15))
axes[1].legend(fontsize=8)
axes[2].set(xticks=positions, xticklabels=labels, ylabel='Target log-odds: +30° minus −30°', title='Does a 30° push change the answer?')
axes[2].axhline(0, color='#555555', linewidth=1)
axes[2].legend(handles=[Line2D([], [], marker='o', linestyle='none', color='#555555', label='Feature direction: one dot per anchor'),
                        Line2D([], [], marker='x', linestyle='none', color='#a3a3a3', label='Two seeded random directions')], fontsize=8)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
fig.suptitle('Direction checks before any FSEC sweep', fontsize=15, y=1.01)
fig.text(.5, -.025, 'Block 8 is one exploratory revision using fresh test items. Each test has 8 matched contents/anchors; dots are not independent replications.', ha='center', fontsize=9)
fig.tight_layout(w_pad=2.3)
for extension in ['png', 'pdf']:
    fig.savefig(OUT / f'validation.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
print(OUT / 'validation.png')

large = json.loads((OUT / 'v2-large-angle/causal-summary.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, target in zip(axes, ['sentiment', 'language']):
    names = [target, 'language' if target == 'sentiment' else 'sentiment', 'random_101', 'random_202']
    for x, name in enumerate(names):
        values = large[target]['directions'][name]['values']
        color = colors[target] if name == target else '#a5a5a5'
        ax.bar(x, np.median(values), color=color, width=.65, alpha=.6)
        ax.scatter(x + np.linspace(-.16, .16, len(values)), values, color=color, s=22)
    ax.axhline(0, color='#444444', linewidth=1)
    ax.set(xticks=range(4), xticklabels=['Candidate\n' + target, 'Other\nfeature', 'Random\n101', 'Random\n202'],
           title='Preference for ' + ('praise' if target == 'sentiment' else 'Spanish'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Change in target log-odds: +60° minus −60°')
fig.suptitle('Larger nudges: recognition still does not establish clean steering', fontsize=13)
fig.text(.5, -.015, 'Bars: medians. Dots: eight anchors. Positive values indicate the intended change. Exploratory follow-up on the same block-8 arrows.', ha='center', fontsize=8)
fig.tight_layout()
for extension in ['png', 'pdf']:
    fig.savefig(OUT / f'large-angle-validation.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
