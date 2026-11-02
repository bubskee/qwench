"""Audit saved matching, response and error records using NumPy (no torch)."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(out):
    prepared = json.loads((out / 'prepared.json').read_text())
    complete = json.loads((out / 'COMPLETE.json').read_text())
    protocol = json.loads((ROOT / 'configs/squared-recovery-v1.json').read_text())
    implementation = json.loads((out / 'implementation-checks.json').read_text())
    assert complete['all_runtime_assertions_passed'] and implementation['all_assertions_passed']
    assert sha(ROOT / 'scripts/check_squared_recovery.py') == implementation['source_sha256']
    assert sha(out / 'prepared.json') == complete['prepared_sha256'] == implementation['prepared_sha256']
    assert sha(ROOT / 'scripts/squared_recovery.py') == complete['source_sha256'] == prepared['source_sha256']
    assert sha(ROOT / 'configs/squared-recovery-v1.json') == prepared['protocol_sha256']
    assert sha(out / 'models.npz') == prepared['models_sha256']
    assert sha(out / 'inputs.npz') == prepared['inputs_sha256']
    source = ROOT / 'reports' / protocol['source_report']
    assert sha(source / 'weights-100000.safetensors') == prepared['model_weights_sha256']
    geometry = json.loads((out / 'geometry.json').read_text())
    corruption = json.loads((out / 'corruption.json').read_text())
    assert sha(out / 'geometry.npz') == geometry['raw_sha256']
    assert sha(out / 'corruption-errors.npz') == corruption['raw_sha256']
    with np.load(out / 'geometry.npz') as data:
        assert data['l2'].shape == (12, 4, 41, 41)
        assert np.isfinite(data['l2']).all() and (data['l2'] >= 0).all()
        assert float(data['l2'][..., 0, 0].max()) == 0
    with np.load(out / 'corruption-errors.npz') as raw:
        for value in raw.values():
            assert value.shape == (2048,) and np.isfinite(value).all() and (value >= 0).all()
        for level_index, sigma in enumerate(protocol['corruption']['coordinate_noise_std']):
            for row in (r for r in corruption['records'] if r['sigma'] == sigma):
                prefix = f'{level_index}_{row["model"]}_{row["method"]}__'
                for metric in ('all', 'full', 'active', 'absent', 'negative'):
                    mean = raw[prefix + metric + '_sum'].sum() / raw[prefix + metric + '_count'].sum()
                    assert abs(mean - row[metric + '_mse']) < 1e-12
                assert np.array_equal(raw[prefix + 'all_count'], raw[prefix + 'active_count'] + raw[prefix + 'absent_count'] + raw[prefix + 'negative_count'])
                assert np.max(np.abs(raw[prefix + 'all_sum'] - sum(raw[prefix + key + '_sum'] for key in ('active', 'absent', 'negative')))) < 1e-10
        with np.load(out / 'inputs.npz') as inputs, np.load(out / 'models.npz') as models:
            supported = inputs['supported']; clean = inputs['clean_x']; target = np.maximum(clean, 0)
            sigma = protocol['corruption']['primary_std']; corrupted = clean[None] + sigma * inputs['standard_noise']
            # Independent scalar formula bypasses the hidden-layer implementation.
            coefficients = ((np.maximum(models['independent_gate'].T, 0) ** 2) @ models['independent_out'].T).diagonal()[supported]
            predicted = np.maximum(corrupted[..., supported], 0) ** 2 * coefficients
            squared = (predicted - target[:, supported]) ** 2
            primary_index = protocol['corruption']['coordinate_noise_std'].index(sigma)
            prefix = f'{primary_index}_independent_natural_squared_relu__'
            assert np.max(np.abs(squared.sum(-1).mean(0) - raw[prefix + 'all_sum'])) < 1e-10
            for label, mask in [('active', clean[:, supported] > 0), ('absent', clean[:, supported] == 0)]:
                assert np.max(np.abs((squared * mask).sum(-1).mean(0) - raw[prefix + label + '_sum'])) < 1e-10
    amendment = json.loads((out / 'source-amendment.json').read_text()) if (out / 'source-amendment.json').exists() else None
    identical_retry = sha(out / 'geometry.npz') == amendment['failed_attempt_geometry_sha256'] if amendment else None
    record = {'all_assertions_passed': True, 'geometry_surfaces': 12,
              'recovery_records': len(corruption['records']), 'independent_scalar_formula_verified': True,
              'all_error_sums_counts_and_pooled_metrics_consistent': True,
              'geometry_identical_after_serialization_fix': identical_retry,
              'primary_feature_fits_flagged': sum(bool(row['fits']['0.5']['flags']) for row in geometry['records'] if row['kind'] == 'encoder_columns'),
              'primary_random_fits_flagged': sum(bool(row['fits']['0.5']['flags']) for row in geometry['records'] if row['kind'] == 'gram_matched_random'),
              'source_sha256': sha(Path(__file__))}
    (out / 'quality-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/squared-recovery-v1')
    main(parser.parse_args().out.resolve())
