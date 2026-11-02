"""Numerical correctness checks for formula-written neural-network weights."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import characterization_core as C
import coupled_characterization as K
import train_scalar_additions as T
import write_toy_weights as W
import evaluate_written_toys as E


class WrittenToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_axis_selector_for_arbitrary_signed_inputs(self):
        x = np.random.default_rng(42).normal(size=(1000, 100)).astype(np.float32)
        model = {k: v.astype(np.float32) for k, v in W.axis_selector().items()}
        expected = np.maximum(x, 0); expected[:, 50:] = 0
        np.testing.assert_array_equal(E.predict(x, model, 'relu'), expected)

    def test_sparse_code_algebra_and_negative_rays(self):
        for ones in [5, 10]:
            model = W.sparse_code(ones)
            np.testing.assert_array_equal(model['gate'].sum(0), np.full(100, ones))
            np.testing.assert_allclose(np.diag(model['out'] @ model['gate']), .9, atol=1e-12)
            self.assertEqual(np.linalg.matrix_rank(model['out']), 49)
            self.assertTrue(np.all(model['gate'] >= 0))
            model = {k: v.astype(np.float32) for k, v in model.items()}
            np.testing.assert_array_equal(E.predict(-np.eye(100, dtype=np.float32), model, 'relu'), 0)
            np.testing.assert_array_equal(E.predict(np.eye(100, dtype=np.float32), model, 'relu').argmax(1), np.arange(100))

    def test_all_clean_bit_patterns_and_both_contexts(self):
        bits = ((np.arange(65536)[:, None] >> np.arange(16)) & 1).astype(np.float32)
        for core in [C, K]:
            a = core.encoder()
            for context_enabled in [False, True]:
                task = 'context_clean' if context_enabled else 'linear_clean'
                model = {k: v.astype(np.float32) for k, v in W.binary_decoder(a, context_enabled).items()}
                for sign in [-1., 1.]:
                    context = np.full(65536, sign, dtype=np.float32)
                    x = C.task_inputs(torch.from_numpy(bits), torch.from_numpy(context), torch.tensor(a, dtype=torch.float32), task).numpy()
                    np.testing.assert_allclose(E.predict(x, model, 'relu'), bits, atol=3e-6, rtol=0)

    def test_denoising_formula_independent_of_network_forward(self):
        for core in [C, K]:
            a = core.encoder(); variance = .35**2 * np.diag(np.linalg.inv(a.T @ a))
            midpoint = .5 + variance * np.log(7)
            # Check continuous values, context routing and joint multi-feature input.
            rng = np.random.default_rng(77)
            decoded = rng.uniform(-1, 2, (2000, 16)).astype(np.float32)
            context = rng.choice([-1., 1.], 2000).astype(np.float32)
            for context_enabled in [False, True]:
                task = 'context_denoise' if context_enabled else 'linear_denoise'
                x = C.task_inputs(torch.from_numpy(decoded), torch.from_numpy(context), torch.tensor(a, dtype=torch.float32), task).numpy()
                model = {k: v.astype(np.float32) for k, v in W.binary_decoder(a, context_enabled, True).items()}
                logits = (decoded - midpoint) / variance
                knots, values = ([-2, 2], [0, 1]) if context_enabled else ([-3, -1, 1, 3], [0, .25, .75, 1])
                expected = np.interp(logits, knots, values)
                np.testing.assert_allclose(E.predict(x, model, 'relu'), expected, atol=5e-6, rtol=0)

    def test_existing_metric_parity(self):
        rng = np.random.default_rng(91)
        x = ((rng.random((8192, 100)) < .02) * rng.uniform(-1, 1, (8192, 100))).astype(np.float32)
        model = {k: v.astype(np.float32) for k, v in W.sparse_code().items()}
        group = {k: torch.tensor(np.repeat(v[None], 6, axis=0)) for k, v in model.items()}
        with patch.dict(T.P, {'architectures': [{'name': 'test', 'activation': 'relu', 'width': 50, 'parameters': 10000}]}):
            old = T.evaluate({'test': group}, torch.from_numpy(x))[0]
        new = E.bottleneck_metrics(x, E.predict(x, model, 'relu'), model, 'relu')
        self.assertEqual(old['competent_feature_indices'], new['competent_feature_indices'])
        for old_key, new_key in [('mean_per_feature_positive_mse', 'positive_mse'),
                                 ('inactive_rmse', 'inactive_rmse'), ('negative_rmse', 'negative_rmse'),
                                 ('isolated_active_mse', 'isolated_active_mse'),
                                 ('multi_input_active_mse', 'multi_input_active_mse'),
                                 ('mean_competent_feature_participation', 'mean_competent_feature_participation'),
                                 ('mean_neuron_feature_participation', 'mean_neuron_feature_participation')]:
            self.assertAlmostEqual(old[old_key], new[new_key], delta=2e-6)

    def test_serialized_architecture_and_round_trip(self):
        models, _ = W.build_models()
        for model in models.values():
            inputs = model['gate'].shape[1]
            expected = 10000 if inputs == 100 else 3216
            self.assertEqual(sum(v.size for v in model.values()), expected)
            self.assertTrue(all(v.dtype == np.float32 and np.isfinite(v).all() for v in model.values()))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'weights'
            W.save_models(out)
            loaded = load_file(out / 'weights.safetensors')
            for name, model in models.items():
                for role, value in model.items():
                    np.testing.assert_array_equal(value, loaded[f'{name}.{role}'])
            with self.assertRaises(FileExistsError): W.save_models(out)


if __name__ == '__main__': unittest.main()
