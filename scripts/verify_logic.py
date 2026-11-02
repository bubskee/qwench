"""Seal or verify logical-pilot artifacts with stdlib SHA-256 checks."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/toy-logic-v1')
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args(); out = args.out.resolve(); manifest = out / 'manifest.json'
    assert out.is_relative_to(ROOT)
    audit = json.loads((out / 'quality-audit.json').read_text())
    controls = json.loads((out / 'projection-check.json').read_text())
    assert audit['all_assertions_passed'] and controls['all_assertions_passed']
    if args.seal:
        prepared = json.loads((out / 'prepared.json').read_text())
        paths = {ROOT / relative for relative in prepared['dependencies']['sources']}
        for relative, expected in prepared['dependencies']['sources'].items():
            assert sha(ROOT / relative) == expected
        for relative in ['scripts/logic_controls.py', 'scripts/plot_logic.py', 'scripts/run_logic.py',
                         'scripts/verify_logic.py', 'configs/toy-logic-controls-v1.json',
                         'requirements/toy.txt', 'requirements/figures.txt',
                         'requirements-toy.txt', 'requirements-reproduction.txt']:
            paths.add(ROOT / relative)
        paths.update(p for p in out.rglob('*') if p.is_file() and p != manifest
                     and 'training-state' not in p.relative_to(out).parts and p.name != '.DS_Store')
        files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(paths)}
        value = {'algorithm': 'sha256', 'paths_relative_to': 'repository root',
                 'scope': 'Logical-pilot primary/secondary artifacts and sources; optimizer scratch and this manifest excluded.',
                 'files': files}
        manifest.write_text(json.dumps(value, indent=2) + '\n')
    saved = json.loads(manifest.read_text())
    for relative, expected in saved['files'].items():
        path = (ROOT / relative).resolve(); assert path.is_relative_to(ROOT) and path.is_file(), relative
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], relative
    print('PASS', len(saved['files']), 'SHA-256 entries; saved primary and independent score audits passed.')
    print('Integrity check; this does not repeat training or numerical audits.')


if __name__ == '__main__': main()
