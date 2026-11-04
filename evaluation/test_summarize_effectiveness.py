"""Synthetic report regression tests; no image data, prediction or training.

Run with ``python -m unittest evaluation.test_summarize_effectiveness`` in the
REPA-E environment. Plot tests use the evaluator's existing matplotlib/numpy
dependencies and write only to a temporary directory.
"""
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from evaluation import summarize_results as reporting
except ModuleNotFoundError as error:
    if not error.name.startswith(('matplotlib', 'numpy')):
        raise
    reporting = None
    IMPORT_ERROR = str(error)
else:
    IMPORT_ERROR = ''


@unittest.skipIf(reporting is None, f'Report plotting dependencies unavailable: {IMPORT_ERROR}')
class EffectivenessReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='repae-report-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = {}
        self.indices = [0, 1]
        self.indices_hash = hashlib.sha256(json.dumps(self.indices).encode()).hexdigest()
        for method in reporting.METHODS:
            directory = self.root / method
            directory.mkdir()
            self.runs[method] = directory
            self.write_json(directory / 'config.json', self.config(method))
            self.write_json(directory / 'indices.json', dict(
                indices=self.indices, split='test', split_csv_sha256='a' * 64))
            self.write_rows(directory, self.rows())

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    @staticmethod
    def write_rows(directory, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def config(self, method):
        arguments = dict(
            weights='ema', split='test', num_samples=2, num_steps=50, seed=1234,
            cfg_scale=1., heun=False, evaluations=['identity', 'effectiveness'],
            identity_metrics=['mae', 'mse', 'ssim'], tasks=['reconstruction', 'counterfactual'],
            attributes=['view', 'age'], target_view=None, target_density=None,
            target_scanner=None, target_age=None, age_bin_edges=[0, 40, 60, 101],
            feature_model='dinov2_vitb14')
        predictor = dict(
            checkpoint_sha256='b' * 64, state_dict_sha256='c' * 64,
            checkpoint_path=f'/mounted/on/{method}/attribute_predictor.pt',
            backbone='resnet50', resolution=[256, 192], selected_epoch=3,
            schema={'view': {'kind': 'categorical', 'classes': 2},
                    'density': {'kind': 'categorical', 'classes': 4},
                    'scanner': {'kind': 'categorical', 'classes': 5},
                    'age': {'kind': 'continuous', 'unit': 'years'}},
            split_csv_sha256={'train': 'd' * 64, 'valid': 'e' * 64, 'test': 'a' * 64},
            test_metrics={'n': 2, 'view': {'accuracy': .9, 'balanced_accuracy': .9, 'macro_f1': .9},
                          'density': {'accuracy': .8, 'balanced_accuracy': .75, 'macro_f1': .75},
                          'scanner': {'accuracy': .85, 'balanced_accuracy': .8, 'macro_f1': .8},
                          'age': {'mae_years': 4., 'rmse_years': 5.}})
        config = dict(
            status='complete', checkpoint=f'/synthetic/{method}/step.pt', experiment=method,
            step=100, resolution=[256, 192], split='test', split_csv_sha256='a' * 64,
            indices_sha256=self.indices_hash, num_images=2, weights='ema',
            inference='synthetic saved-image fixture', feature_extractors={},
            source_sha256={}, image_storage={'format': 'synthetic uint16 grayscale PNG'},
            sampling_source_sha256={}, reference_images_sha256='f' * 64,
            arguments=arguments, attribute_predictor=predictor, age_metric='mae_years')
        self.assertTrue(set(reporting.PROTOCOL_METADATA).issubset(config))
        self.assertTrue(set(reporting.PROTOCOL_ARGUMENTS).issubset(arguments))
        return config

    @staticmethod
    def row(comparison, attribute, metric, value, target='all', family='effectiveness'):
        return dict(family=family, comparison=comparison, attribute=attribute,
                    target=target, metric=metric, value=value, std='', n_real=2,
                    n_generated=2, status='ok', low_sample_count='', rank_deficient='')

    def rows(self):
        return [
            self.row('cf_target', 'view', 'target_accuracy', .75),
            self.row('cf_target', 'view', 'balanced_target_accuracy', .7),
            self.row('input_target', 'view', 'target_accuracy', .25),
            self.row('cf_target', 'view', 'target_accuracy', .75, target='1'),
            self.row('input_target', 'view', 'target_accuracy', .25, target='1'),
            self.row('cf_target', 'age', 'target_mae_years', 3.5),
            self.row('input_target', 'age', 'target_mae_years', 10.),
            self.row('cf_target', 'age', 'target_mae_years', 3.5, target='60'),
            self.row('real_prediction', 'view', 'accuracy', .9),
            self.row('real_prediction', 'view', 'balanced_accuracy', .9),
            self.row('real_prediction', 'age', 'mae_years', 4.),
            self.row('vae_vs_input', 'none', 'mae', .01, family='identity'),
            self.row('cycle_vs_input', 'none', 'mae', .02, family='identity'),
            self.row('cf_vs_input', 'view', 'mae', .03, family='identity'),
            self.row('cf_vs_cycle', 'view', 'mae', .02, family='identity'),
        ]

    def change_config(self, method, update):
        path = self.runs[method] / 'config.json'
        config = json.loads(path.read_text())
        update(config)
        self.write_json(path, config)

    def test_identical_fingerprints_accept_different_checkpoint_paths(self):
        protocol, selection, fingerprints = reporting.validate_results(self.runs)
        self.assertEqual(selection['indices'], self.indices)
        self.assertEqual(set(fingerprints), set(reporting.METHODS))
        self.assertEqual(protocol['attribute_predictor']['checkpoint_sha256'], 'b' * 64)
        self.assertNotIn('checkpoint_path', protocol['attribute_predictor'])

    def test_reject_different_checkpoint_or_state_fingerprints(self):
        for key in ('checkpoint_sha256', 'state_dict_sha256'):
            with self.subTest(key=key):
                self.change_config('REPA', lambda config: config['attribute_predictor'].update({key: '1' * 64}))
                with self.assertRaisesRegex(ValueError, 'attribute predictor'):
                    reporting.validate_results(self.runs)
                self.write_json(self.runs['REPA'] / 'config.json', self.config('REPA'))

    def test_reject_unsupported_age_metric(self):
        self.change_config('REPA-E', lambda config: config.update(age_metric='rmse_years'))
        with self.assertRaisesRegex(ValueError, 'age metric'):
            reporting.validate_results(self.runs)

    def test_legacy_age_tolerances_do_not_change_mae_comparability(self):
        self.change_config('REPA', lambda config: config.pop('age_metric'))
        self.change_config('LDM', lambda config: config['arguments'].update(age_tolerance_years=5.))
        self.change_config('REPA-E', lambda config: config['arguments'].update(age_tolerance_years=2.))
        protocol, _, _ = reporting.validate_results(self.runs)
        self.assertEqual(protocol['age_metric'], 'mae_years')

    def test_reject_different_counterfactual_target_assignments(self):
        for method in reporting.METHODS:
            self.change_config(method, lambda config: config.update(
                counterfactual_target_policy='uniform_v1', counterfactual_targets_sha256='2' * 64,
                age_range_years=[30., 90.], training_age_csv_sha256='3' * 64))
        protocol, _, _ = reporting.validate_results(self.runs)
        self.assertEqual(protocol['counterfactual_targets_sha256'], '2' * 64)
        self.change_config('REPA', lambda config: config.update(counterfactual_targets_sha256='4' * 64))
        with self.assertRaisesRegex(ValueError, 'target assignments|target protocol'):
            reporting.validate_results(self.runs)

    def test_reject_missing_predictor_metadata(self):
        self.change_config('REPA', lambda config: config.pop('attribute_predictor'))
        with self.assertRaisesRegex(ValueError, 'attribute_predictor metadata'):
            reporting.validate_results(self.runs)

    def test_existing_non_effectiveness_reports_remain_valid(self):
        for method, directory in self.runs.items():
            config = self.config(method)
            config['arguments']['evaluations'] = ['identity']
            config.pop('age_metric')
            config.pop('attribute_predictor')
            self.write_json(directory / 'config.json', config)
            self.write_rows(directory, [row for row in self.rows() if row['family'] == 'identity'])
        protocol, _, _ = reporting.validate_results(self.runs)
        self.assertNotIn('attribute_predictor', protocol)
        self.assertNotIn('age_tolerance_years', protocol['settings'])

    def test_effectiveness_tables_label_separate_metric_directions(self):
        rows = [dict(row, method='LDM') for row in self.rows()]
        table = reporting._effectiveness_table(rows, ['LDM'], ['view', 'age'])
        for heading in ('Target accuracy ↑', 'Balanced target accuracy ↑',
                        'Target MAE (years) ↓'):
            self.assertIn(heading, table)
        self.assertIn('0.75', table)
        self.assertIn('3.5', table)
        self.assertIn('Input target accuracy', table)
        self.assertIn('Input target MAE (years)', table)
        self.assertNotIn('RMSE', table)
        self.assertNotIn('tolerance', table)

    def test_legacy_age_metrics_are_omitted_from_report_tables(self):
        rows = [dict(row, method='LDM') for row in self.rows()]
        rows.extend(dict(row, method='LDM') for row in [
            self.row('cf_target', 'age', 'target_rmse_years', 4.5),
            self.row('cf_target', 'age', 'target_within_tolerance', .5),
            self.row('real_prediction', 'age', 'rmse_years', 5.),
        ])
        table = reporting._effectiveness_table(rows, ['LDM'], ['view', 'age'])
        diagnostics = reporting._predictor_diagnostics(
            rows, ['LDM'], self.config('LDM')['attribute_predictor'])
        for text in (table, diagnostics):
            self.assertNotIn('RMSE', text)
            self.assertNotIn('rmse_years', text)
            self.assertNotIn('tolerance', text)
        self.assertIn('mae_years', diagnostics)

    def test_build_report_renders_synthetic_effectiveness_png_and_pdf(self):
        protocol, _, _ = reporting.validate_results(self.runs)
        protocol['synthetic_preview'] = True
        destination = self.root / 'report'
        artifacts = reporting.build_report(destination, self.runs, protocol)
        report = (destination / 'report.md').read_text()
        self.assertIn('SYNTHETIC PREVIEW', report)
        self.assertIn('## Counterfactual effectiveness', report)
        self.assertIn('held-out real test patients', report)
        self.assertIn('cached factual images', report)
        self.assertIn('mean absolute error', report)
        self.assertIn('Target MAE (years) ↓', report)
        self.assertNotIn('RMSE', report)
        self.assertNotIn('tolerance', report)
        self.assertIn('Checkpoint SHA256', report)
        self.assertIn('b' * 64, report)
        self.assertIn('Balanced target accuracy ↑', report)
        for extension in ('png', 'pdf'):
            path = destination / 'figures' / f'counterfactual_effectiveness.{extension}'
            self.assertIn(path, artifacts)
            payload = path.read_bytes()
            self.assertGreater(len(payload), 1000)
            if extension == 'png':
                self.assertEqual(payload[:8], b'\x89PNG\r\n\x1a\n')
                width, height = struct.unpack('>II', payload[16:24])
                self.assertGreater(width, 100)
                self.assertGreater(height, 100)
            else:
                self.assertTrue(payload.startswith(b'%PDF-'))
        with (destination / 'effectiveness_targets.csv').open(newline='') as handle:
            target_rows = list(csv.DictReader(handle))
        self.assertEqual(len(target_rows), 9)
        self.assertTrue(all(row['target'] != 'all' for row in target_rows))
        self.assertEqual({row['method'] for row in target_rows}, set(reporting.METHODS))
        self.assertTrue(all(path.is_file() for path in artifacts))


if __name__ == '__main__':
    unittest.main()
