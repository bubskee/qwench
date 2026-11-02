"""Independently check the separable null and geometry measurement machinery."""
import json
import numpy as np
import torch
import characterization_core as C
import analyze_characterization as A
from check_characterization import numpy_forward


def main():
    plan = json.loads((C.OUT / 'analysis-plan.json').read_text())
    with np.load(C.OUT / 'analysis-inputs.npz') as data: assets = {key: data[key].copy() for key in data.files}
    rng = np.random.default_rng(55201)
    records = []
    for track in ['new', 'historical']:
        for name in C.P['activations']:
            candidates = [row for row in plan['networks'] if row['track'] == track and row['activation'] == name
                          and row['seed'] == 0 and row['task'] == ('context_denoise' if track == 'new' else 'L2')
                          and (C.ROOT / row['evaluation']).exists()]
            meta = max(candidates, key=lambda row: row['step'])
            model = A.model_from(meta); maximum = 0.; optimized_error = 0.
            n = 16 if track == 'new' else 100
            for sign in [-1., 1.]:
                for feature in [0, 3, n - 1]:
                    amplitudes = np.array([-.5, 0, .25, 1, 1.5])
                    if track == 'new':
                        encoder = C.effective_encoder(assets['encoder'], meta['task'], sign)
                        x = np.c_[amplitudes[:, None] * encoder[:, feature], np.full(5, sign)]
                    else:
                        x = np.zeros((5, 100)); x[:, feature] = amplitudes
                    natural = numpy_forward(x, model, name)[:, feature]
                    null = A.separable(torch.tensor(x), model, meta, assets['encoder']).numpy()[:, feature]
                    maximum = max(maximum, float(np.abs(natural - null).max()))
            for sign in [-1., 1.]:
                if track == 'new':
                    encoder = C.effective_encoder(assets['encoder'], meta['task'], sign)
                    x = np.c_[rng.normal(size=(4, 32)), np.full(4, sign)]
                    u = x[:, :32] @ encoder
                    expected = []
                    for feature in range(n):
                        probes = np.c_[u[:, feature, None] * encoder[:, feature], np.full(4, sign)]
                        expected.append(numpy_forward(probes, model, name)[:, feature])
                    pair = [0, 1]; delta = np.c_[encoder[:, pair].T, np.zeros(2)]
                    base = np.c_[np.zeros((1, 32)), np.full(1, sign)]
                else:
                    x = rng.normal(size=(4, 100)); expected = []
                    for feature in range(n):
                        probes = np.zeros((4, 100)); probes[:, feature] = x[:, feature]
                        expected.append(numpy_forward(probes, model, name)[:, feature])
                    pair = [83, 94]; delta = np.eye(100)[pair]; base = np.zeros((1, 100))
                actual = A.separable(torch.tensor(x), model, meta, assets['encoder']).numpy()
                maximum = max(maximum, float(np.abs(np.array(expected).T - actual).max()))
                changed = base + np.array([[.3, .7], [.1, -.2]]) @ delta
                full = A.separable(torch.tensor(changed), model, meta, assets['encoder']) - A.separable(torch.tensor(base), model, meta, assets['encoder'])
                partial = A.separable(torch.tensor(changed), model, meta, assets['encoder'], pair) - A.separable(torch.tensor(base), model, meta, assets['encoder'], pair)
                optimized_error = max(optimized_error, float((full.norm(dim=-1) - partial.norm(dim=-1)).abs().max()))
            assert max(maximum, optimized_error) < 1e-10
            records.append({'track': track, 'activation': name, 'tested_network': meta['key'], 'null_vs_independent_numpy_probe_max_error': maximum,
                            'two_output_geometry_optimization_max_error': optimized_error})
    axis = np.linspace(0, 1.5, 31); x, y = np.meshgrid(axis, axis, indexing='ij')
    calibrations = []
    for label, grid, expected in [('linear', np.hypot(x, y), 2.), ('independent_square', np.hypot(x*x, y*y), 4.)]:
        fit, _ = A.FIT.fit(grid, .5)
        assert abs(fit['p'] - expected) < .04
        calibrations.append({'case': label, 'expected_p': expected, 'fit': fit})
    tilted = np.sqrt(x*x - x*y + y*y)
    fit, _ = A.FIT.fit(tilted, .5)
    assert fit['quadratic']['mean_radial_frac'] < .001
    C.write(C.OUT / 'analysis-implementation-checks.json', {'all_assertions_passed': True,
            'separable_checks': records, 'calibrations': calibrations, 'tilted_quadratic_fit': fit,
            'source_sha256': C.sha(__file__), 'analysis_source_sha256': C.sha(A.__file__),
            'input_sha256': C.sha(C.OUT / 'analysis-inputs.npz')})
    print('PASS separable restrictions, independent NumPy implementation, optimized norms and analytic geometry calibration.')


if __name__ == '__main__': main()
