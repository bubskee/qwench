"""Prepare or resume a small AND/XOR reproduction; outputs stay under work/."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('work/reproductions/toy-logic-v1'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    assert out.is_relative_to((ROOT / 'work').resolve()) and out != (ROOT / 'work').resolve(), '--out must be below work/'
    if args.resume:
        assert (out / 'prepared.json').exists(), 'Not a prepared logical-task workspace.'
    else:
        assert not out.exists(), 'Choose a new --out or use --resume.'
        out.mkdir(parents=True)
    env = os.environ.copy(); env['QWENCH_LOGIC_OUT'] = str(out)
    env.update(OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    def run(script, *extra):
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *extra], cwd=ROOT, env=env, check=True)
    if not args.resume:
        subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0,"scripts"); import logic_core as C; C.make_assets()'],
                       cwd=ROOT, env=env, check=True)
    else:
        subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0,"scripts"); import logic_core as C; C.assert_seal()'],
                       cwd=ROOT, env=env, check=True)
    if args.prepare_only:
        print('Prepared frozen inputs; no training launched:', out); return
    if not (out / 'preflight-checks.json').exists(): run('check_logic.py', '--stage', 'preflight')
    if not (out / 'TRAINING-COMPLETE.json').exists():
        checkpoint = out / 'training-state/resume.pt'
        if (out / 'training.jsonl').exists() and not checkpoint.exists():
            raise RuntimeError('Interrupted before first checkpoint; choose a new --out.')
        run('train_logic.py', *(['--resume'] if checkpoint.exists() else []))
    if not (out / 'ANALYSIS-COMPLETE.json').exists(): run('analyze_logic.py')
    run('check_logic.py', '--stage', 'audit'); run('logic_controls.py')
    run('plot_logic.py', '--source', str(out))
    print('Training, analysis and checks complete:', out)


if __name__ == '__main__': main()
