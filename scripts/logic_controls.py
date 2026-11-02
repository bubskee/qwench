"""Final-only projection check and independent Boolean-label score audit."""
import json
import numpy as np
import analyze_logic as A
import logic_core as C


def independent_score(prediction, bits, task):
    left = bits[:, 0::2].astype(bool); right = bits[:, 1::2].astype(bool)
    labels = np.logical_and(left, right) if task.startswith('and') else np.logical_xor(left, right)
    target = np.broadcast_to(labels, prediction.shape)
    error = np.square(prediction.astype(float) - target)
    positive = int(target.sum()); negative = int(target.size - positive)
    pos = float(np.where(target, error, 0).sum() / positive)
    neg = float(np.where(~target, error, 0).sum() / negative)
    return {'all_mse': float(error.sum() / error.size), 'positive_mse': pos, 'negative_mse': neg,
            'balanced_mse': (pos + neg) / 2,
            'positive_miss_rate': float(np.logical_and(prediction < .5, target).sum() / positive),
            'negative_false_positive_rate': float(np.logical_and(prediction >= .5, ~target).sum() / negative),
            'positive_count': positive, 'negative_count': negative}


def main():
    C.assert_seal(); assert (C.OUT / 'TRAINING-COMPLETE.json').exists()
    assets = C.load_assets(); encoder = assets['encoder']; projection = encoder @ encoder.T
    bits = assets['recovery_bits'].astype(float); clean = bits @ encoder.T
    records = []; raw_predictions = []; invariance_error = 0.; independent_error = 0.; audited_frames = 0
    for path in sorted((C.OUT / 'analysis').glob('*_t*.json')):
        record = json.loads(path.read_text()); task = record['network']['task']
        with np.load(path.with_suffix('.npz')) as raw:
            for li, sigma in enumerate(C.P['recovery']['noise_std']):
                for which in ['natural', 'clipped_natural']:
                    pred = raw['recovery_predictions'][li].astype(float)
                    if which == 'clipped_natural': pred = np.clip(pred, 0, 1)
                    expected = independent_score(pred, bits, task)
                    row = next(r for r in record['recovery'] if r['map'] == which and r['sigma'] == sigma)
                    independent_error = max(independent_error, max(abs(row[key] - value) for key, value in expected.items()))
        audited_frames += 1
    assert audited_frames == 180 and independent_error < 1e-12
    for meta in C.network_rows(C.P['training']['steps']):
        model = A.model_from(meta); predict = lambda x: C.numpy_forward(x, model, meta['activation'])
        stored = []
        for sigma in [0., .35]:
            x = clean[None] + sigma * assets['recovery_noise']
            pred = predict(x.reshape(-1, 32) @ projection).reshape(2, len(bits), 8).astype(np.float32)
            for which, prediction in [('projected_natural', pred.astype(float)),
                                      ('projected_clipped_natural', np.clip(pred.astype(float), 0, 1))]:
                score = independent_score(prediction, bits, meta['task'])
                records.append({'network': meta, 'sigma': sigma, 'map': which, **score})
            stored.append(pred)
        raw_predictions.append(np.stack(stored))
        for pair in C.P['geometry']['pairs']:
            z = assets['background_bits'][3].copy(); z[pair['bits']] = 0
            axis = np.linspace(0, 1, 31); aa, bb = np.meshgrid(axis, axis, indexing='ij')
            x = z @ encoder.T + np.stack([aa.ravel(), bb.ravel()], -1) @ encoder[:, pair['bits']].T
            invariance_error = max(invariance_error, float(np.abs(predict(x) - predict(x @ projection)).max()))
    assert invariance_error < 1e-10
    raw_path = C.OUT / 'projection-predictions.npz'
    np.savez_compressed(raw_path, predictions=np.stack(raw_predictions))
    with np.load(raw_path) as raw:
        for mi, meta in enumerate(C.network_rows(30000)):
            for li, sigma in enumerate([0., .35]):
                for which in ['projected_natural', 'projected_clipped_natural']:
                    pred = raw['predictions'][mi, li].astype(float)
                    if which.endswith('clipped_natural'): pred = np.clip(pred, 0, 1)
                    expected = independent_score(pred, bits, meta['task'])
                    row = next(r for r in records if r['network']['key'] == meta['key'] and r['sigma'] == sigma and r['map'] == which)
                    assert all(row[key] == value for key, value in expected.items())
    C.write(C.OUT / 'projection-check.json', {'all_assertions_passed': True, 'records': records,
            'primary_frames_independently_audited': audited_frames,
            'Boolean_label_and_sum_count_audit_max_error': independent_error,
            'feature_plane_projection_invariance_max_error': invariance_error,
            'dependencies': C.dependencies(), 'source_sha256': C.sha(__file__),
            'protocol_sha256': C.sha(C.ROOT / 'configs/toy-logic-controls-v1.json'),
            'raw_sha256': C.sha(raw_path),
            'scope': 'Secondary final-only fairness check; primary training/geometry unchanged.'})
    print('PROJECTION_AND_INDEPENDENT_SCORE_AUDIT_PASS', 'invariance error', invariance_error,
          'score error', independent_error, flush=True)


if __name__ == '__main__': main()
