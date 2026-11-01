"""Compact summary of cached-data audit, separate from original paper renders."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/paper-audit'
data = json.loads((OUT / 'summary.json').read_text())
conic = json.loads((OUT / 'conic-audit.json').read_text())
families = ['Contrastive', 'MELBO', 'SAE', 'PCA', 'Random-diff', 'Random']
colors = ['#327b62'] * 3 + ['#727989'] * 3
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={'width_ratios': [1.05, 1.2, 1.15]})
ys = np.arange(len(families))
medians = [data['families'][key]['median'] for key in families]
axes[0].axvline(2, color='#b36b37', linestyle='--', linewidth=1.3)
axes[0].scatter(medians, ys, c=colors, s=65, zorder=3)
for y, p, key in zip(ys, medians, families):
    axes[0].text(p + .012, y, f'{p:.3f}  (n={data["families"][key]["n"]})', va='center', fontsize=8)
axes[0].set(yticks=ys, yticklabels=families, xlim=(1.92, 2.49), xlabel='Median fitted p', title='Candidate families differ from controls')
axes[0].invert_yaxis()
rotations = sorted((float(theta), value) for theta, value in data['rotation'].items())
theta = np.array([r[0] for r in rotations])
median = np.array([r[1]['median'] for r in rotations])
lo = np.array([r[1]['q25'] for r in rotations])
hi = np.array([r[1]['q75'] for r in rotations])
axes[1].fill_between(theta, lo, hi, color='#367d9c', alpha=.15)
axes[1].plot(theta, median, 'o-', color='#367d9c')
axes[1].axhline(2, color='#b36b37', linestyle='--', linewidth=1.3)
axes[1].set(xlabel='Rotation toward random directions (degrees)', ylabel='Fitted p (median and IQR)', xticks=[0, 30, 60, 90], title='Misalignment reduces the effect')
fractions = [conic['families'][key]['superellipse_wins'] / conic['families'][key]['n'] for key in families]
axes[2].barh(ys, fractions, color=colors, alpha=.85)
axes[2].axvline(.5, color='#777777', linestyle='--', linewidth=1.1)
axes[2].set(yticks=ys, yticklabels=families, xlim=(0, 1), xlabel='Fraction of pairs with lower residual', title='Superellipse vs tilted ellipse')
axes[2].invert_yaxis()
for y, fraction in zip(ys, fractions):
    axes[2].text(fraction + .02, y, f'{fraction:.0%}', va='center', fontsize=9)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.14)
    ax.set_axisbelow(True)
fig.suptitle('FSEC cached-data audit: the main patterns reproduce', fontsize=15, y=1.02)
fig.text(.5, -.035, 'Gemma-2-9B supplied caches; 18 sampled raw grids refit. Right panel is a later repository analysis. This is not a fresh LLM replication.', ha='center', fontsize=9)
fig.tight_layout(w_pad=2.3)
for extension in ['png', 'pdf']:
    fig.savefig(OUT / f'audit-summary.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
print(OUT / 'audit-summary.png')
