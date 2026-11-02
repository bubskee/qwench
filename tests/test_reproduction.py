"""Checks for protecting evidence and resuming the frozen trainer correctly."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'reproduce', Path(__file__).resolve().parents[1] / 'scripts/reproduce.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root_patch = patch.object(R, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def seed_repository(self):
        for relative in ('scripts/train_scalar_core.py', 'configs/example.json',
                         'vendor/license', 'reports/toy-cis-v1/prepared.json',
                         f'reports/{R.COHORTS[0]}/geometry-inputs.npz',
                         f'reports/{R.COHORTS[0]}/geometry-inputs.json'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'frozen-input\n')

    def test_outputs_cannot_target_evidence_or_work_root(self):
        for path in ('reports/new', 'scripts/new', 'work', '../escape'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                R.output_path(path, 'toy-figures')
        self.assertEqual(R.output_path(None, 'toy-figures'),
                         self.root / 'work/reproductions/toy-figures')

    def test_symlink_cannot_escape_work(self):
        (self.root / 'work').mkdir()
        (self.root / 'reports').mkdir()
        (self.root / 'work/link').symlink_to(self.root / 'reports', target_is_directory=True)
        with self.assertRaises(ValueError):
            R.output_path('work/link/overwrite', 'toy-figures')

    def test_existing_output_is_never_overwritten(self):
        path = self.root / 'work/existing'
        path.mkdir(parents=True)
        (path / 'evidence').write_text('retain me')
        with self.assertRaises(ValueError):
            R.create_output(path)
        self.assertEqual((path / 'evidence').read_text(), 'retain me')

    def test_preparation_and_resume_preserve_source_integrity(self):
        self.seed_repository()
        out = self.root / 'work/new-training'
        with patch.object(R, 'run') as run:
            R.train_toys(out, prepare_only=True)
            R.train_toys(out, resume=True, prepare_only=True)
            run.assert_not_called()
        staged = out / 'outputs/qwench'
        self.assertTrue((out / 'work').is_dir())
        state = json.loads((out / 'run.json').read_text())
        for relative, digest in state['source_files_sha256'].items():
            self.assertEqual(R.sha(self.root / relative), digest)
            self.assertEqual(R.sha(staged / relative), digest)
        (staged / 'scripts/train_scalar_core.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source/input changed'):
            R.train_toys(out, resume=True, prepare_only=True)
        self.assertEqual((self.root / 'scripts/train_scalar_core.py').read_bytes(), b'frozen-input\n')

    def test_resume_rolls_back_uncheckpointed_log_without_losing_it(self):
        log = self.root / 'training.jsonl'
        rows = [{'step': step} for step in [1, *range(1000, 13000, 1000)]]
        log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        R.rewind_training_log(log, 10000)
        self.assertEqual([json.loads(line)['step'] for line in log.read_text().splitlines()],
                         [1, *range(1000, 11000, 1000)])
        backup = self.root / 'resume-discarded-1.jsonl'
        self.assertEqual([json.loads(line)['step'] for line in backup.read_text().splitlines()],
                         [11000, 12000])
        R.rewind_training_log(log, 10000)
        self.assertFalse((self.root / 'resume-discarded-2.jsonl').exists())

    def test_missing_checkpoint_log_steps_fail_before_rewriting(self):
        path = self.root / 'training.jsonl'
        original = '{"step": 1}\n{"step": 10000}\n'
        path.write_text(original)
        with self.assertRaisesRegex(ValueError, 'Incomplete/duplicate'):
            R.rewind_training_log(path, 10000)
        self.assertEqual(path.read_text(), original)

    def test_numeric_comparison_allows_roundoff_but_preserves_counts_and_schema(self):
        expected = {'p': [2.4], 'count': 30, 'flagged': False}
        close = {'p': [2.4000001], 'count': 30, 'flagged': False}
        result = R.compare_numerical(close, expected)
        self.assertTrue(result['within_tolerance'])
        self.assertFalse(result['exact_equal'])
        for changed in ({**close, 'count': 31}, {**close, 'p': [2.41]},
                        {**close, 'flagged': 0}, {'p': [2.4], 'count': 30}):
            self.assertFalse(R.compare_numerical(changed, expected)['within_tolerance'])

    def test_full_study_stages_required_history_and_rejects_changed_resume_sources(self):
        self.seed_repository()
        (self.root / 'requirements').mkdir()
        (self.root / 'requirements/figures.txt').write_text('numpy\n')
        for relative in ('reports/toy-scalar-comparison-v1/geometry-inputs.npz',
                         'reports/toy-scalar-comparison-v1/geometry-inputs.json'):
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'frozen')
        for cohort in R.COHORTS:
            for step in [10000, 30000, 100000]:
                for name in [f'weights-{step:06d}.safetensors', f'eval-{step:06d}.json']:
                    path = self.root / 'reports' / cohort / name
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'frozen')
        out = self.root / 'work/study'
        with patch.object(R, 'run') as run:
            R.study_run(out, prepare_only=True)
            R.study_run(out, resume=True, prepare_only=True)
            run.assert_not_called()
        state = json.loads((out / 'run.json').read_text())
        self.assertEqual(state['fresh_networks'], 120)
        self.assertEqual(state['checkpoint_frames'], 810)
        staged = out / 'outputs/qwench'
        for relative, digest in state['source_files_sha256'].items():
            self.assertEqual(R.sha(staged / relative), digest)
        (staged / 'configs/example.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source/input changed'):
            R.study_run(out, resume=True, prepare_only=True)


if __name__ == '__main__':
    unittest.main()
