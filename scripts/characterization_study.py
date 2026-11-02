"""Ordered full-study pipeline for a separately prepared reproduction workspace.

This script uses the frozen trainers and measurements, skips completed stages,
and makes no Git changes. Prefer reproduce.py study-run to prepare the workspace.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    command = [sys.executable, str(ROOT / 'scripts' / script), *args]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    import characterization_core as C
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('Full study training requires native Apple MPS; saved study figures work on CPU.')
    first = C.OUT; second = ROOT / 'reports/toy-characterization-coupled-v1'
    if not (first / 'assets.npz').exists(): C.make_assets()
    if not (first / 'preflight-checks.json').exists(): run('check_characterization.py')
    if not (first / 'TRAINING-COMPLETE.json').exists():
        checkpoint = C.WORK / 'resume.pt'
        if (first / 'training.jsonl').exists() and not checkpoint.exists():
            raise RuntimeError('Interrupted before the first training checkpoint; choose a new study output.')
        run('train_characterization.py', *(['--resume'] if checkpoint.exists() else []))
    if not (first / 'analysis-plan.json').exists(): run('analyze_characterization.py', '--stage', 'prepare')
    if not (first / 'analysis-implementation-checks.json').exists(): run('check_characterization_analysis.py')
    for track in ['new', 'historical']:
        if not (first / f'analysis-complete-{track}.json').exists():
            run('analyze_characterization.py', '--stage', 'measure', '--track', track)
    if not (second / 'assets.npz').exists(): run('coupled_characterization.py', '--stage', 'prepare')
    if not (second / 'preflight-checks.json').exists(): run('coupled_characterization.py', '--stage', 'check')
    if not (second / 'TRAINING-COMPLETE.json').exists():
        checkpoint = ROOT / 'work/toy-characterization-coupled-v1/resume.pt'
        if (second / 'training.jsonl').exists() and not checkpoint.exists():
            raise RuntimeError('Coupled training stopped before its first checkpoint; choose a new output.')
        run('coupled_characterization.py', '--stage', 'train', *(['--resume'] if checkpoint.exists() else []))
    if not (second / 'analysis-plan.json').exists(): run('coupled_characterization.py', '--stage', 'analysis-prepare')
    if not (second / 'analysis-complete-new.json').exists(): run('coupled_characterization.py', '--stage', 'measure')
    for family in ['orthogonal', 'coupled']:
        if not (ROOT / f'reports/toy-characterization-diagnostics-v1/{family}.json').exists():
            run('diagnose_characterization.py', '--family', family)
        if not (ROOT / f'reports/toy-characterization-axis-null-v1/{family}.json').exists():
            run('axis_null_characterization.py', '--family', family)
        if not (ROOT / f'reports/toy-characterization-pair-diagnostic-v1/{family}.json').exists():
            run('pair_diagnostic_characterization.py', '--family', family)
    if not (ROOT / 'reports/toy-characterization-oracle-geometry-v1/analysis.json').exists():
        run('oracle_geometry_characterization.py')
    if not (ROOT / 'reports/toy-characterization-validation-v1/inputs.npz').exists():
        run('validate_characterization.py', '--stage', 'prepare')
    for family in ['orthogonal', 'coupled']:
        if not (ROOT / f'reports/toy-characterization-validation-v1/{family}.json').exists():
            run('validate_characterization.py', '--stage', 'measure', '--family', family)
        run('audit_characterization.py', '--family', family)
    run('summarize_characterization.py'); run('plot_characterization.py')
    run('audit_characterization_study.py')
    C.write(ROOT / 'reports/toy-characterization-study-v1/PIPELINE-COMPLETE.json',
            {'all_stages_finished': True, 'fresh_networks': 120, 'checkpoint_frames': 810,
             'meaning': 'Fresh new training plus saved historical checkpoints; no new LLM inference.'})


if __name__ == '__main__': main()
