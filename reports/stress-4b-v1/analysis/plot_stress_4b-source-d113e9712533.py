"""Full-cohort diagnostic plots for the sealed Qwen stress run (no fit filtering)."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from safetensors.numpy import load_file
import analyze_stress_4b as A

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'paper': '#317b67', 'novel': '#3875a4', 'control': '#82838a'}


def save(fig, path):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + ext), dpi=155, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def normalized_contour(record, run, anchor_indices=None):
    data = np.load(run / 'surfaces' / (record['id'] + '.npz'))
    grid = data['l2'] if anchor_indices is None else data['l2'][anchor_indices]
    angles, median = data['angles_deg'], np.median(grid, axis=0)
    t1, t2 = record['headline'].get('intercepts', [None, None])
    if t1 is None or t2 is None:
        return None
    points = A.F.extract_contour(angles, median, record['headline']['threshold'])
    points = points[(points[:, 0] < angles.max() - 1) & (points[:, 1] < angles.max() - 1)]
    radius = np.hypot(points[:, 0], points[:, 1])
    safe = np.where(radius > 1e-12, radius, 1.)
    return np.sin(np.deg2rad(radius))[:, None] * points / safe[:, None] / np.sin(np.deg2rad([t1, t2]))


def gallery(primary, run, out):
    ncols = 5; nrows = (len(primary) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, primary):
        p = r['headline']['p']; points = normalized_contour(r, run)
        if points is not None:
            ax.plot(points[:, 0], points[:, 1], color=COLORS[r['kind']], linewidth=1.6)
        circle_x, circle_y = A.F.superellipse_curve(2)
        ax.plot(circle_x, circle_y, color='#999999', linestyle=':', linewidth=1)
        if p is not None:
            x, y = A.F.superellipse_curve(p)
            ax.plot(x, y, color='#c17238', linestyle='--', linewidth=1)
        name = r['a'] + ' × ' + r['b'] if r['kind'] != 'control' else r['stratum'].replace('_', ' ') + '\nseed ' + r['a'].split('_')[-2]
        score = 'no fit' if p is None else f'p={p:.3f}, residual={100*r["headline"]["mean_radial_frac"]:.2f}%'
        ax.set(title=name + '\n' + score + (' [flag]' if r['headline'].get('flags') else ''),
               xlim=(0, 1.2), ylim=(0, 1.2), aspect='equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1]); ax.tick_params(labelsize=8)
        ax.title.set_size(9)
    for ax in axes.flat[len(primary):]:
        ax.axis('off')
    fig.suptitle('Every completed primary pair: measured boundary, fitted p (dashed), p=2 (dotted)\nExact sine-geodesic coordinates normalized by axis crossings; each pair has its own threshold.', fontsize=13, y=1.005)
    save(fig, out / 'all-contours')


def eligibility(plan, run, out):
    names = sorted(plan['feature_metadata'], key=lambda n: (plan['feature_metadata'][n]['family'], n))
    directions = load_file(str(run / 'directions.safetensors'))
    mat = np.stack([directions[n] for n in names]); cos = mat @ mat.T
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    image = axes[0].imshow(cos, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(image, ax=axes[0], fraction=.04, label='Raw cosine overlap')
    axes[1].imshow(np.abs(cos) < .1, cmap='Greys', vmin=0, vmax=1)
    for ax in axes:
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
        ax.tick_params(axis='x', rotation=90, labelsize=7); ax.tick_params(axis='y', labelsize=7)
        families = [plan['feature_metadata'][n]['family'] for n in names]
        for i in range(1, len(names)):
            if families[i] != families[i-1]:
                ax.axhline(i-.5, color='#cf752a', linewidth=1); ax.axvline(i-.5, color='#cf752a', linewidth=1)
    axes[0].set_title('Before any response measurements: 37 candidate directions')
    axes[1].set_title('Paper overlap filter: black cells pass |cosine| < 0.1')
    fig.suptitle('Pair eligibility is itself model-dependent; diagonals are never tested', fontsize=14)
    save(fig, out / 'eligibility')


def anchor_plot(primary, plan, out):
    feature_rows = sorted([r for r in primary if r['kind'] != 'control'], key=lambda r: (r['kind'], r['id']))
    if not feature_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, .27 * len(feature_rows))), gridspec_kw={'width_ratios':[1.8, 1]})
    for i, r in enumerate(feature_rows):
        values = [v['own_threshold_fit']['p'] for v in r['per_anchor']]
        for j, p in enumerate(values):
            if p is not None:
                axes[0].scatter(p, i + (j-3.5)*.045, c=[plt.cm.tab10(j)], s=22, alpha=.85)
        if r['headline']['p'] is not None:
            axes[0].scatter(r['headline']['p'], i, marker='D', color='black', s=25, zorder=4)
        for j, factor in enumerate(['0.5', '1.0', '2.0']):
            p = r['threshold_sensitivity'][factor]['p']
            if p is not None:
                axes[1].scatter(p, i + (j-1)*.12, color=['#3875a4', '#222222', '#bc713d'][j], s=22)
    for ax in axes:
        ax.axvline(2, color='#999999', linestyle='--', linewidth=1)
        ax.set(yticks=range(len(feature_rows)), ylim=(len(feature_rows)-.5, -.5), xlabel='Fitted p')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=.18)
    axes[0].set_yticklabels([r['a'] + ' × ' + r['b'] for r in feature_rows], fontsize=8)
    axes[1].set_yticklabels([])
    axes[0].set_title('Eight individual anchors (own threshold); black diamond = median surface')
    axes[1].set_title('Threshold sensitivity of median surface')
    axes[1].legend([Line2D([0],[0], marker='o', color=c, linestyle='') for c in ['#3875a4','#222222','#bc713d']],
                   ['0.5 × T', '1 × T', '2 × T'], loc='best', fontsize=8)
    labels = [f'{i}: {a["decoded_window"]!r}' for i, a in enumerate(plan['primary_anchors'])]
    fig.text(.01, -.025, '\n'.join(labels), fontsize=8)
    save(fig, out / 'anchor-and-threshold-sensitivity')


def paired_controls(primary, plan, out):
    by_pair = {r['pair_id']: r for r in primary}
    pairs = []
    for row in primary:
        if row['stratum'] != 'same_prompt_random_relabelling':
            continue
        m = plan['control_metadata']
        sources = {m[row[k]]['source_feature'] for k in ['a', 'b']}
        feature = next((r for r in primary if {r['a'],r['b']} == sources), None)
        if feature is not None:
            pairs.append((feature, row))
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
    for ax, (feature, control) in zip(axes.flat, pairs):
        for i in range(len(feature['per_anchor'])):
            ps = [r['per_anchor'][i]['own_threshold_fit']['p'] for r in [feature, control]]
            if all(p is not None for p in ps):
                ax.plot([0,1], ps, '-o', color=plt.cm.tab10(i), alpha=.65, markersize=4, linewidth=1)
        ax.plot([0,1], [r['headline']['p'] for r in [feature, control]], '-D', color='black', linewidth=2, markersize=6)
        ax.axhline(2, color='#999999', linestyle=':')
        ax.set(xticks=[0,1],xticklabels=['True labels','Balanced random labels'], ylabel='Fitted p', title=feature['a'] + ' × ' + feature['b'])
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same training prompts, changed labels: colors = individual anchors; black = median surface', fontsize=12)
    save(fig, out / 'same-prompt-control')


def diagnostics(rows, run, out):
    primary = {r['pair_id']: r for r in rows if r['tag']=='primary'}
    diagnostic = [r for r in rows if r['tag']!='primary']
    if not diagnostic:
        return
    fig, ax = plt.subplots(figsize=(13, max(5, .35*len(diagnostic))))
    for i, r in enumerate(diagnostic):
        p, q = primary[r['pair_id']]['headline']['p'], r['headline']['p']
        # Sign runs use only the first four anchors, so compute that same subset baseline.
        if r['tag'].startswith('signs'):
            d = np.load(run/'surfaces'/(r['pair_id']+'_primary.npz'))
            median = np.median(d['l2'][:4],axis=0)
            p = A.fit(d['angles_deg'],median,A.threshold(median))['p']
        if p is not None:
            ax.scatter(p,i,color='#888888',s=22)
        if q is not None:
            ax.scatter(q,i,color=COLORS[r['kind']],marker='D',s=30)
        if p is not None and q is not None:
            ax.plot([p,q],[i,i],color='#aaaaaa',linewidth=1)
    ax.set(yticks=range(len(diagnostic)),yticklabels=[r['a']+' × '+r['b']+' / '+r['tag'] for r in diagnostic],xlabel='Fitted p',ylim=(len(diagnostic)-.5,-.5))
    ax.tick_params(axis='y',labelsize=8); ax.axvline(2,color='#bc713d',linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Preselected diagnostics: circles = matched primary baseline; diamonds = changed condition\nSign comparisons use the same four anchors; other diagnostics use all eight English anchors as baseline.')
    save(fig,out/'fixed-diagnostics')


def sign_geometry(rows, run, out):
    primary = {r['pair_id']: r for r in rows if r['tag'] == 'primary'}
    by_pair = {}
    for r in rows:
        if r['tag'].startswith('signs'):
            by_pair.setdefault(r['pair_id'], []).append(r)
    ready = {key: values for key, values in by_pair.items() if len(values) == 3}
    if not ready:
        return
    fig, axes = plt.subplots(1, len(ready), figsize=(5 * len(ready), 5), squeeze=False)
    for ax, (key, variants) in zip(axes.flat, ready.items()):
        p0 = primary[key]
        data = np.load(run / 'surfaces' / (p0['id'] + '.npz'))
        median = np.median(data['l2'][:4], axis=0)
        matched = {**p0, 'headline': A.fit(data['angles_deg'], median, A.threshold(median))}
        shared_threshold = A.threshold(median)
        fixed_variants = []
        for r in variants:
            d = np.load(run / 'surfaces' / (r['id'] + '.npz'))
            fixed = A.fit(d['angles_deg'], np.median(d['l2'], axis=0), shared_threshold)
            fixed_variants.append(({**r, 'headline': fixed}, None))
        for r, indices in [(matched, list(range(4)))] + fixed_variants:
            points = normalized_contour(r, run, indices)
            signs = np.array(r['signs'])
            if points is not None:
                points = points * signs
                ax.plot(points[:, 0], points[:, 1], color='#317b67', linewidth=2)
            p = r['headline']['p']
            label = 'no fit' if p is None else f'p={p:.2f}'
            ax.text(.65 * signs[0], .68 * signs[1], label, ha='center', fontsize=10,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
        theta = np.linspace(0,2*np.pi,300)
        ax.plot(np.cos(theta),np.sin(theta),':',color='#999999',linewidth=1)
        ax.axhline(0,color='#bbbbbb',linewidth=.6); ax.axvline(0,color='#bbbbbb',linewidth=.6)
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',xlabel=p0['a']+' direction',ylabel=p0['b']+' direction',title=p0['a']+' × '+p0['b'])
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Changing the signs of the push: identical first four English anchors and shared L2 threshold\nThreshold from the original quadrant; each axis normalized by its crossing at that threshold.',fontsize=12)
    save(fig,out/'sign-geometry')


def response_traces(rows, run, out):
    selected_ids = ['feature_000_primary', 'feature_001_primary', 'feature_006_primary', 'control_033_primary']
    by_id = {r['id']: r for r in rows}
    selected = [by_id[key] for key in selected_ids if key in by_id]
    if not selected:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    for ax, r in zip(axes.flat, selected):
        data = np.load(run / 'surfaces' / (r['id'] + '.npz'))
        angles = data['angles_deg']; grid = np.median(data['l2'], axis=0)
        ax.plot(angles, grid[:, 0], '-o', markersize=3, color='#317b67', label='Pure first direction')
        ax.plot(angles, grid[0, :], '-o', markersize=3, color='#3875a4', label='Pure second direction')
        radius = np.sqrt(2) * angles; keep = radius <= 60
        ax.plot(radius[keep], np.diag(grid)[keep], '-o', markersize=3, color='#9a659b', label='Equal mixture')
        ax.axhline(r['headline']['threshold'], color='#bc713d', linestyle='--', label='Pair threshold')
        name = r['a'] + ' × ' + r['b'] if r['kind'] != 'control' else 'Isotropic control, seed 101'
        ax.set(xlabel='Total angular distance of push (degrees)', ylabel='Median downstream residual L2 change', title=name, xlim=(0,60))
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15)
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Raw sensitivity traces for the three fixed challenge pairs and first random control\nAxes have different strengths; the fitted p additionally normalizes by their threshold crossings.',fontsize=12)
    save(fig,out/'response-traces')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/stress-4b-v1')
    args=parser.parse_args(); run=args.run.resolve(); out=run/'analysis'
    plan=json.loads((run/'plan.json').read_text())
    rows=[json.loads((out/(t['id']+'.json')).read_text()) for t in plan['tasks'] if (out/(t['id']+'.json')).exists()]
    primary=[r for r in rows if r['tag']=='primary']
    eligibility(plan,run,out)
    if primary:
        gallery(primary,run,out); anchor_plot(primary,plan,out); paired_controls(primary,plan,out)
    diagnostics(rows,run,out)
    sign_geometry(rows,run,out)
    response_traces(rows,run,out)
    print('PLOTTED',len(rows),'surfaces')

if __name__=='__main__':
    main()
