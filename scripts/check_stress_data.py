"""Check sealed stress artifacts and calibrate the fitter on known geometry."""
import hashlib
import json
from pathlib import Path
import numpy as np
import analyze_stress_4b as A

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/stress-4b-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    seal = json.loads((RUN / 'input-seal.json').read_text())
    assert plan['seal'] == seal
    assert sha(ROOT / 'configs/stress-4b-v1.json') == seal['protocol_sha256']
    assert sha(ROOT / 'data/stress-features-v1.json') == seal['feature_data_sha256']
    protocol = json.loads((ROOT / 'configs/stress-4b-v1.json').read_text())
    assert sha(Path(protocol['corpus_path'])) == protocol['corpus_sha256']
    assert sha(Path(protocol['corpus_provenance_path'])) == protocol['corpus_provenance_sha256']
    anchor_rows = {a['corpus_row'] for a in plan['primary_anchors']}
    assert not anchor_rows & set(plan['control_pool_corpus_rows'])
    assert len(anchor_rows) == 8 and len(plan['control_pool_corpus_rows']) == 248
    records, examples = [], 0
    for task in plan['tasks']:
        path = RUN / 'surfaces' / (task['id'] + '.npz')
        if not path.exists():
            continue
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['seal'] == seal and metadata['task'] == task
        source = RUN / ('sweep-source-' + metadata['source_sha256'][:12] + '.py')
        assert sha(source) == metadata['source_sha256']
        data = np.load(path, allow_pickle=False)
        n = len(task['angles_deg'])
        assert data['l2'].shape == (len(task['anchor_bank']), n, n)
        assert np.array_equal(data['angles_deg'], task['angles_deg'])
        assert np.isfinite(data['l2']).all() and (data['l2'] >= 0).all()
        assert np.all(data['l2'][:, 0, 0] == 0)
        assert metadata['zero_angle_l2_max'] == 0
        assert metadata['max_relative_norm_error'] < .005
        analysis = RUN / 'analysis' / (task['id'] + '.json')
        if analysis.exists():
            assert json.loads(analysis.read_text())['raw_sha256'] == sha(path)
        records.append(metadata); examples += data['l2'].size
    assert not plan['feature_names_uncovered']
    assert len(plan['feature_names_covered']) == 37
    assert all(v == 0 for v in plan['capture_validation_max_abs_error'].values())
    # Known p contours at the same angular spacing; no model or fitted constants.
    calibrations = []
    for step in [5., 2.5]:
        angles = np.arange(0., 60. + step / 2, step)
        a, b = np.meshgrid(angles, angles, indexing='ij')
        radius = np.hypot(a, b); safe = np.where(radius > 0, radius, 1.)
        x = np.sin(np.deg2rad(radius)) * a / safe / np.sin(np.deg2rad(42.))
        y = np.sin(np.deg2rad(radius)) * b / safe / np.sin(np.deg2rad(48.))
        for p in [1.5, 2., 2.5, 3.]:
            surface = (x**p + y**p)**(1/p)
            fit = A.fit(angles, surface, 1.)
            error = fit['p'] - p
            assert abs(error) < .06, (step, p, fit['p'])
            calibrations.append({'spacing_deg': step, 'known_p': p, 'recovered_p': fit['p'], 'error': error})
    complete = (RUN / 'COMPLETE.json').exists()
    if complete:
        assert len(records) == len(plan['tasks'])
        assert examples == plan['planned_examples']
    overlap = []
    for task in plan['tasks']:
        if task['tag'] not in ['fine_grid', 'fp32']:
            continue
        variant_path = RUN / 'surfaces' / (task['id'] + '.npz')
        if not variant_path.exists():
            continue
        variant = np.load(variant_path)
        primary = np.load(RUN / 'surfaces' / (task['pair_id'] + '_primary.npz'))
        indices = [int(np.flatnonzero(variant['angles_deg'] == x)[0]) for x in primary['angles_deg']]
        changed = variant['l2'][:, indices][:, :, indices]
        reference = primary['l2']
        delta = np.abs(changed - reference)
        reference_median = np.median(reference, axis=0)
        changed_median = np.median(changed, axis=0)
        reference_fit = A.fit(primary['angles_deg'], reference_median, A.threshold(reference_median))
        resampled_fit = A.fit(primary['angles_deg'], changed_median, A.threshold(changed_median))
        overlap.append({'task': task['id'], 'matched_points': int(reference.size),
                        'max_absolute_l2_difference': float(delta.max()),
                        'mean_absolute_l2_difference': float(delta.mean()),
                        'relative_mean_absolute_l2_difference': float(delta.mean() / reference.mean()),
                        'primary_grid_p': reference_fit['p'],
                        'variant_at_primary_grid_points_p': resampled_fit['p'],
                        'note': 'For fine grids, compare this subsampled p with the full fine-grid p to separate shared-point response changes from contour resolution.'})
    result = {'complete': complete, 'checked_surfaces': len(records), 'planned_surfaces': len(plan['tasks']),
              'checked_examples': examples, 'planned_examples': plan['planned_examples'],
              'zero_angle_l2_max': max((r['zero_angle_l2_max'] for r in records), default=0),
              'max_relative_norm_error': max((r['max_relative_norm_error'] for r in records), default=0),
              'all_artifact_assertions_passed': True, 'synthetic_fitter_calibration': calibrations,
              'refinement_and_precision_response_comparisons': overlap,
              'note': 'Artifact and numerical checks do not validate the semantic purity of candidate directions.'}
    A.write(RUN / 'analysis' / 'quality-audit.json', result)
    print(json.dumps(A.clean(result), indent=2))

if __name__ == '__main__':
    main()
