"""Seal or verify logical-pilot artifacts with stdlib SHA-256 checks."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def derived_check(out):
    rows = json.loads((out / 'networks.json').read_text()); assert len(rows) == 180
    controls = json.loads((out / 'projection-check.json').read_text())['records']
    for row in rows:
        source = json.loads((out / 'analysis' / (row['key'] + '.json')).read_text())
        assert all(row[key] == value for key, value in source['network'].items())
        for which in ['natural', 'clipped_natural', 'plugin_reference', 'Bayes_oracle', 'constant_prior']:
            score = next(s for s in source['recovery'] if s['map'] == which and s['sigma'] == .35)
            for metric in ['all_mse', 'positive_mse', 'negative_mse', 'balanced_mse', 'positive_miss_rate', 'negative_false_positive_rate']:
                assert row[which + '_' + metric] == score[metric]
        clean = next(s for s in source['recovery'] if s['map'] == 'natural' and s['sigma'] == 0.)
        assert row['clean_mse'] == clean['all_mse'] and row['clean_balanced_mse'] == clean['balanced_mse']
        assert row['noise_penalty'] == row['natural_all_mse'] - row['clean_mse']
        for pair in ['within', 'across']:
            case = next(c for c in source['geometry'] if c['pair']['name'] == pair and c['state'] == [0, 0])
            assert row[pair + '_p'] == case['backgrounds'][3]['fit']['p']
            assert row[pair + '_flags'] == case['backgrounds'][3]['fit']['all_flags']
        for metric in ['all_mse', 'positive_mse', 'negative_mse', 'balanced_mse']:
            actual = row['projected_clipped_' + metric]
            if row['step'] == 30000:
                expected = next(c for c in controls if c['network']['key'] == row['key'] and c['sigma'] == .35 and c['map'] == 'projected_clipped_natural')[metric]
                assert actual == expected
            else: assert actual is None
    groups = json.loads((out / 'seed-summary.json').read_text()); assert len(groups) == 60
    for group in groups:
        selected = [r for r in rows if all(r[key] == group[key] for key in ['activation', 'task', 'step'])]
        assert len(selected) == group['seeds'] == 3
        for key, summary in group.items():
            if not isinstance(summary, dict): continue
            if key.endswith('_p_good'):
                pair = key.removesuffix('_p_good')
                values = [r[pair + '_p'] for r in selected if r[pair + '_p'] is not None and not r[pair + '_flags']]
                assert summary['count'] == len(values)
                assert summary['mean'] == (statistics.mean(values) if values else None)
            else:
                values = [r[key] for r in selected]
                assert summary == {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    return {'all_assertions_passed': True, 'network_rows': 180, 'seed_groups': 60,
            'meaning': 'Summary fields and seed means checked against primary and secondary saved records.'}


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
        (out / 'summary-checks.json').write_text(json.dumps(derived_check(out), indent=2) + '\n')
        prepared = json.loads((out / 'prepared.json').read_text())
        assert audit['dependencies'] == prepared['dependencies'] and controls['dependencies'] == prepared['dependencies']
        assert sha(out / 'assets.npz') == prepared['dependencies']['assets_sha256']
        assert controls['source_sha256'] == sha(ROOT / 'scripts/logic_controls.py')
        assert controls['protocol_sha256'] == sha(ROOT / 'configs/toy-logic-controls-v1.json')
        plotted = json.loads((out / 'plot-receipt.json').read_text())
        assert plotted['plot_source_sha256'] == sha(ROOT / 'scripts/plot_logic.py')
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
