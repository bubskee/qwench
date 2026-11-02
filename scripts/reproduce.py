"""Portable entry points around the original, hash-sealed experiment scripts.

Run --help for commands. New outputs stay under work/; reports/ are evidence.
This file uses only the standard library. Figure/refit dependencies are loaded
by child processes; full training additionally requires the original MPS stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOY = 'toy-scalar-comparison-v1'
COHORTS = ('toy-scalars-core-v1', 'toy-scalar-additions-v1')
MANIFESTS = ('toy-trained-v1', *COHORTS, TOY)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def output_path(value, command):
    """Reserve new outputs below work without touching evidence or source files."""
    path = Path(value) if value else ROOT / 'work/reproductions' / command
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    work = (ROOT / 'work').resolve()
    if not path.is_relative_to(work) or path == work:
        raise ValueError('--out must be a new subdirectory of this clone\'s work/')
    return path


def create_output(path):
    if path.exists():
        raise ValueError(f'Output already exists: {path}. Choose a new --out directory.')
    path.mkdir(parents=True)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run(script, *args, root=ROOT, env=None):
    command = [sys.executable, str(root / 'scripts' / script), *map(str, args)]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def verify():
    count = 0
    for name in MANIFESTS:
        manifest = ROOT / 'reports' / name / 'manifest.json'
        files = json.loads(manifest.read_text())['files']
        for relative, expected in files.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f'Missing/invalid artifact: {relative}')
            if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError(f'Artifact differs from manifest: {relative}')
            count += 1
        print(f'PASS {name}: {len(files)} file entries', flush=True)
    audit = json.loads((ROOT / 'reports' / TOY / 'quality-audit.json').read_text())
    if not audit['all_assertions_passed'] or audit['trained_networks'] != 30:
        raise ValueError('Saved numerical audit is incomplete')
    print(f'PASS {count} manifest entries; saved audit covers 30 completed networks.')
    print('This verifies saved artifacts, not a new numerical or training run.')


def compare_numerical(actual, expected):
    """Keep schemas/counts exact; allow small floating-point backend differences."""
    mismatches = []
    largest = 0.0

    def visit(left, right, path):
        nonlocal largest
        if isinstance(right, dict):
            if not isinstance(left, dict) or left.keys() != right.keys():
                mismatches.append(path + ': object keys differ')
                return
            for key in right:
                visit(left[key], right[key], path + '.' + key)
        elif isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                mismatches.append(path + ': list length/type differs')
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f'{path}[{index}]')
        elif isinstance(right, float):
            if not isinstance(left, (int, float)) or isinstance(left, bool):
                mismatches.append(path + ': numeric type differs')
                return
            if not math.isfinite(left) or not math.isfinite(right):
                mismatches.append(path + ': nonfinite number')
                return
            largest = max(largest, abs(left - right))
            if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8):
                mismatches.append(path + ': number outside tolerance')
        elif type(left) is not type(right) or left != right:
            mismatches.append(path + ': value differs')

    visit(actual, expected, '$')
    return {'exact_equal': actual == expected, 'within_tolerance': not mismatches,
            'float_rtol': 1e-6, 'float_atol': 1e-8,
            'max_abs_numeric_difference': largest, 'mismatches': mismatches[:20]}


def toy_analysis(out, refit=False):
    source = ROOT / 'reports' / TOY
    create_output(out)
    for path in sorted(source.glob('eval-*.json')):
        copy_file(path, out / path.name)
    if refit:
        for name in ('GEOMETRY-COMPLETE.json', 'geometry-plan.json'):
            copy_file(source / name, out / name)
        shutil.copytree(source / 'surfaces', out / 'surfaces')
    else:
        copy_file(source / 'analysis/cohorts.json', out / 'analysis/cohorts.json')
    env = os.environ.copy()
    env.update(QWENCH_TRAINED_DRIVER='reproduction_runtime', QWENCH_REPRO_OUT=str(out),
               MPLCONFIGDIR=str(out / '.matplotlib'))
    if refit:
        run('fit_trained_toys.py', env=env)
    run('plot_trained_toys.py', env=env)
    # Compare numerical products, not PNG bytes (rendering metadata may differ).
    products = ('model-summary.json',) + (('analysis/cohorts.json',) if refit else ())
    comparisons = {}
    for name in products:
        comparisons[name] = compare_numerical(json.loads((out / name).read_text()),
                                             json.loads((source / name).read_text()))
    write(out / 'reproduction-check.json', {
        'mode': 'refit_saved_surfaces' if refit else 'plot_saved_fits',
        'source': f'reports/{TOY}', 'numerical_comparisons': comparisons,
        'source_plot_sha256': sha(ROOT / 'scripts/plot_trained_toys.py'),
    })
    if not all(row['within_tolerance'] for row in comparisons.values()):
        raise ValueError(f'Regenerated numerical products differ; inspect {out}')
    print(f'PASS numerical products match saved results within tolerance. Figures: {out}', flush=True)


def prepare_training(out):
    """Copy frozen sources into their expected layout without changing their bytes."""
    create_output(out)
    staged = out / 'outputs/qwench'
    staged.mkdir(parents=True)
    for name in ('scripts', 'configs', 'vendor'):
        shutil.copytree(ROOT / name, staged / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (staged / 'reports' / TOY).mkdir(parents=True)
    (out / 'work').mkdir()
    for name in COHORTS:
        (staged / 'reports' / name).mkdir()
    # These backgrounds and pair ordering were frozen before final measurements.
    for name in ('geometry-inputs.npz', 'geometry-inputs.json'):
        copy_file(ROOT / 'reports' / COHORTS[0] / name, staged / 'reports' / COHORTS[0] / name)
    # The fitter imports this saved metadata; it does not use the released weights.
    copy_file(ROOT / 'reports/toy-cis-v1/prepared.json', staged / 'reports/toy-cis-v1/prepared.json')
    files = {}
    for folder in ('scripts', 'configs', 'vendor'):
        for path in (staged / folder).rglob('*'):
            if path.is_file():
                files[str(path.relative_to(staged))] = sha(path)
    for name in ('geometry-inputs.npz', 'geometry-inputs.json'):
        path = staged / 'reports' / COHORTS[0] / name
        files[str(path.relative_to(staged))] = sha(path)
    path = staged / 'reports/toy-cis-v1/prepared.json'
    files[str(path.relative_to(staged))] = sha(path)
    write(out / 'run.json', {'kind': 'fresh_full_toy_training', 'root': str(staged),
          'source_files_sha256': files, 'networks': 30, 'updates_each': 100000,
          'hardware': 'Apple Silicon MPS, float32',
          'note': 'Both cohorts start from initialization; original interruption is not replayed.'})
    print(f'Prepared byte-identical sources at {staged}', flush=True)
    return staged


def rewind_training_log(path, checkpoint_step):
    """Keep the frozen trainer's audit log consistent with its resumed state."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    retained = [row for row in rows if row['step'] <= checkpoint_step]
    discarded = [row for row in rows if row['step'] > checkpoint_step]
    expected = [1] + list(range(1000, checkpoint_step + 1, 1000))
    if [row['step'] for row in retained] != expected:
        raise ValueError(f'Incomplete/duplicate checkpoint history: {path}')
    if discarded:
        index = 1
        while (path.parent / f'resume-discarded-{index}.jsonl').exists():
            index += 1
        backup = path.parent / f'resume-discarded-{index}.jsonl'
        backup.write_text(''.join(json.dumps(row) + '\n' for row in discarded))
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(row) + '\n' for row in retained))
        temp.replace(path)
        print(f'Resuming at {checkpoint_step}; later log rows preserved in {backup}', flush=True)


def train_toys(out, resume=False, prepare_only=False):
    if resume:
        state = json.loads((out / 'run.json').read_text())
        staged = out / 'outputs/qwench'
        if state['kind'] != 'fresh_full_toy_training' or Path(state['root']) != staged:
            raise ValueError('Not a matching prepared training workspace')
        for relative, expected in state['source_files_sha256'].items():
            if sha(staged / relative) != expected:
                raise ValueError(f'Prepared source/input changed: {relative}')
    else:
        staged = prepare_training(out)
    if prepare_only:
        print(f'No training launched. To execute: {sys.executable} scripts/reproduce.py '
              f'toy-train --out {out} --resume', flush=True)
        return
    import torch
    import safetensors  # fail before training if the complete dependency set is missing
    if not torch.backends.mps.is_available():
        raise ValueError('Full training uses Apple Silicon MPS. Use toy-figures or toy-refit on CPU.')
    env = os.environ.copy()
    env['QWENCH_TRAINED_DRIVER'] = 'scalar_comparison_runtime'
    env.pop('QWENCH_REPRO_OUT', None)
    env['MPLCONFIGDIR'] = str(out / '.matplotlib')
    for cohort, script in zip(COHORTS, ('train_scalar_core.py', 'train_scalar_additions.py')):
        if (staged / 'reports' / cohort / 'TRAINING-COMPLETE.json').exists():
            continue
        checkpoint = out / 'work' / cohort / 'resume.pt'
        log = staged / 'reports' / cohort / 'training.jsonl'
        if log.exists() and not checkpoint.exists():
            raise ValueError(f'{cohort}: interrupted before its first checkpoint; use a new --out.')
        if checkpoint.exists():
            if not log.exists():
                raise ValueError(f'{cohort}: checkpoint exists but its training log is missing.')
            saved_step = torch.load(checkpoint, map_location='cpu', weights_only=True)['step']
            rewind_training_log(log, saved_step)
        run(script, *(['--resume'] if checkpoint.exists() else []), root=staged, env=env)
    combined = staged / 'reports' / TOY
    if not (combined / 'TRAINING-COMPLETE.json').exists():
        run('assemble_scalar_comparison.py', root=staged, env=env)
    run('check_trained_geometry.py', root=staged, env=env)
    if not (combined / 'geometry-plan.json').exists():
        run('geometry_trained_toys.py', '--stage', 'prepare', root=staged, env=env)
    # The plan exists on disk before the first p measurement. No git mutations.
    run('geometry_trained_toys.py', '--stage', 'sweep', root=staged, env=env)
    for script in ('fit_trained_toys.py', 'plot_trained_toys.py',
                   'summarize_scalar_comparison.py', 'audit_trained_toys.py'):
        run(script, root=staged, env=env)
    print(f'Full training and analysis complete: {combined}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify', help='Check saved toy artifacts against SHA-256 manifests (stdlib only)')
    for name, help_text in (
        ('toy-figures', 'Rebuild five toy figures from saved fits; CPU, no torch'),
        ('toy-refit', 'Refit saved toy response grids and rebuild figures; CPU, no torch'),
        ('paper', 'Fetch the pinned paper cache and reproduce Figure 2; CPU, network/git'),
        ('toy-train', 'Train all 30 networks from initialization, then measure geometry; Apple MPS'),
    ):
        child = commands.add_parser(name, help=help_text)
        child.add_argument('--out', help=f'New output under work/ (default: work/reproductions/{name})')
        if name == 'toy-train':
            child.add_argument('--resume', action='store_true', help='Continue a matching prepared workspace')
            child.add_argument('--prepare-only', action='store_true', help='Stage sources/inputs without training')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
        return
    out = output_path(args.out, args.command)
    if args.command in ('toy-figures', 'toy-refit'):
        toy_analysis(out, refit=args.command == 'toy-refit')
    elif args.command == 'paper':
        create_output(out)
        run('reproduce_paper.py', '--reference-dir', ROOT / 'work/fsec-reference', '--out', out)
    else:
        train_toys(out, resume=args.resume, prepare_only=args.prepare_only)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
