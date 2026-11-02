"""Directly coupled versus independent probe pairs, with a full-vector null."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import characterization_core as B
import analyze_characterization as A


def main(family):
    if family == 'orthogonal': C = B; null = A.separable
    else:
        import coupled_characterization as C
        null = C.separable
    output = B.ROOT / 'reports/toy-characterization-pair-diagnostic-v1'
    output.mkdir(exist_ok=True)
    assert not (output / (family + '.json')).exists()
    protocol_path = B.ROOT / 'configs/toy-characterization-pair-diagnostic-v1.json'
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads((C.OUT / 'analysis-plan.json').read_text())
    metas = [row for row in plan['networks'] if row['track'] == 'new' and row['step'] == 100000]
    with np.load(C.OUT / 'analysis-inputs.npz') as raw: assets = dict(raw)
    all_records = []; raw_grids = []; null_grids = []; prior_error = 0.; control_error = 0.
    axis = np.linspace(0, 1.5, 31); aa, bb = np.meshgrid(axis, axis, indexing='ij')
    for meta in metas:
        model = A.model_from(meta)
        previous = json.loads((C.OUT / 'analysis/new' / (meta['key'] + '.json')).read_text())
        with np.load(C.OUT / 'analysis/new' / (meta['key'] + '.npz')) as previous_raw:
            for pair in protocol['pairs']:
                for sign in [[1, 1], [1, -1]]:
                    coefficients = torch.tensor(np.stack([sign[0] * aa.ravel(), sign[1] * bb.ravel()], -1))
                    for which in ['natural', 'separable']:
                        measured = []; predicted = []; vector_residual = []
                        for bg in range(4):
                            context = float(assets['background_context'][bg])
                            encoder = C.effective_encoder(assets['encoder'], meta['task'], context)
                            bits = assets['background_bits'][bg].copy(); bits[pair] = 0
                            base = torch.tensor(np.r_[encoder @ bits, context][None])
                            d = torch.tensor(np.c_[encoder[:, pair].T, np.zeros(2)])
                            def f(x):
                                return C.forward(x, model, meta['activation']) if which == 'natural' else null(x, model, meta, assets['encoder'])
                            y0 = f(base)
                            y1 = f(base + coefficients[:, :1] @ d[:1])
                            y2 = f(base + coefficients[:, 1:] @ d[1:])
                            y12 = f(base + coefficients @ d)
                            vector_prediction = y1 + y2 - y0
                            r = (y12 - y0).norm(dim=-1).numpy().reshape(31, 31)
                            predicted_r = (vector_prediction - y0).norm(dim=-1).numpy().reshape(31, 31)
                            measured.append(r); predicted.append(predicted_r)
                            vector_residual.append(float((y12 - vector_prediction).norm(dim=-1).mean()) / max(float(r.max()), 1e-12))
                        measured = np.stack(measured); predicted = np.stack(predicted)
                        if which == 'separable':
                            error = float(np.abs(measured - predicted).max()); control_error = max(control_error, error)
                            assert error < 1e-8
                        index = next((i for i, row in enumerate(previous['geometry']) if row['pair'] == pair
                                      and row['anchor'] == 0 and row['sign'] == sign and row['kind'] == 'feature_axes' and row['map'] == which), None)
                        if index is not None:
                            error = float(np.abs(measured - previous_raw['response'][index]).max())
                            prior_error = max(prior_error, error); assert error < 1e-8
                        cosine = float(assets['encoder'][:, pair[0]] @ assets['encoder'][:, pair[1]])
                        actual_fits = {str(fraction): A.FIT.fit(np.median(measured, axis=0), fraction)[0] for fraction in [.25, .5, .75]}
                        predicted_fits = {str(fraction): A.FIT.fit(np.median(predicted, axis=0), fraction)[0] for fraction in [.25, .5, .75]}
                        all_records.append({'network': meta, 'family': family, 'pair': pair, 'map': which, 'sign': sign,
                            'input_cosine': cosine, 'original_fits': actual_fits, 'vector_additive_fits': predicted_fits,
                            'vector_interaction_residual_by_background': vector_residual,
                            'mean_vector_interaction_residual': float(np.mean(vector_residual)),
                            'norm_response_residual_fraction': float(np.mean(np.abs(measured - predicted)) / max(measured.max(), 1e-12))})
                        raw_grids.append(measured); null_grids.append(predicted)
        print('PAIR_DIAGNOSTIC', family, meta['key'], flush=True)
    np.savez_compressed(output / (family + '.npz'), response=np.stack(raw_grids), vector_null=np.stack(null_grids))
    B.write(output / (family + '.json'), {'records': all_records, 'source_sha256': B.sha(__file__),
            'protocol_sha256': B.sha(protocol_path), 'raw_sha256': B.sha(output / (family + '.npz')),
            'all_control_additivity_checks_passed': True, 'separable_max_abs_additivity_error': control_error,
            'existing_surface_reproduction_max_abs_error': prior_error})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['orthogonal', 'coupled'], required=True)
    with torch.inference_mode(): main(parser.parse_args().family)
