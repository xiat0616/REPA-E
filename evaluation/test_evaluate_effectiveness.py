"""Saved-image integration tests with artificial pixels/predictions; never train a model."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
import evaluate_all
from sample.sample_io import (IMAGE_STORAGE, counterfactual_targets_hash, reference_images_hash, save_json,
                              write_image, write_records)


class SavedEffectivenessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / 'samples'
        self.cache.mkdir()
        self.indices = [0, 1]
        records = []
        for index in self.indices:
            labels = dict(view=index, density=index + 1, scanner=index + 2, age=40 + index * 20)
            input_path = f'real/{index}.png'
            for task, attribute in [('real', 'none'), ('reconstruction_vae', 'none'),
                                    ('reconstruction_cycle', 'none'),
                                    ('counterfactual', 'view'), ('counterfactual', 'age')]:
                row = dict(labels)
                source, target = '', ''
                if task == 'counterfactual':
                    source = labels[attribute]
                    target = 1 - source if attribute == 'view' else source + 10
                    row[attribute] = target
                path = f'{task}/{attribute}/{index}.png' if task != 'real' else input_path
                digest = write_image(self.cache, path, torch.full((1, 16, 16), .2 + .1 * index))
                records.append(dict(index=index, task=task, attribute=attribute,
                                    source=source, target=target, file_path=path,
                                    sha256=digest, input_path=input_path, **row))
        self.config = dict(
            status='complete', checkpoint='synthetic.pt', step=1, experiment='synthetic',
            weights='ema', vae_source='synthetic', model='synthetic', resolution=[16, 16],
            split='test', split_csv_sha256='a' * 64, num_images=2,
            indices_sha256=hashlib.sha256(json.dumps(self.indices).encode()).hexdigest(),
            image_storage=IMAGE_STORAGE, reference_images_sha256=reference_images_hash(records),
            eligible_interventions={'view': 2, 'age': 2},
            arguments=dict(tasks=['reconstruction', 'counterfactual'], attributes=['view', 'age'], seed=1))
        save_json(self.cache / 'config.json', self.config)
        save_json(self.cache / 'indices.json', dict(split='test', split_csv_sha256='a' * 64, indices=self.indices))
        write_records(self.cache, records)
        self.predictor_metadata = dict(
            resolution=[16, 16], split_csv_sha256={'test': 'a' * 64},
            checkpoint_sha256='b' * 64, state_dict_sha256='c' * 64,
            test_metrics={'n': 2})

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *extra):
        return evaluate_all.parse_args([
            '--samples-dir', str(self.cache), '--evaluations', 'identity', 'effectiveness',
            '--identity-metrics', 'mae', 'mse', 'ssim', '--attribute-predictor', 'synthetic.pt',
            '--age-tolerance-years', '1', '--batch-size', '2', '--device', 'cpu',
            '--cpu-threads', '1', '--output-root', str(self.root / 'results'), *extra])

    def test_saved_cache_scores_requested_targets_and_keeps_existing_metrics(self):
        predictions = [
            dict(view=torch.tensor([[2., 0.], [2., 0.]]), age=torch.tensor([42., 63.])),
            dict(view=torch.tensor([[0., 2.], [0., 2.]]), age=torch.tensor([42., 63.])),
            dict(view=torch.tensor([[2., 0.], [0., 2.]]), age=torch.tensor([48., 70.])),
        ]
        predictor = Mock(side_effect=predictions)
        before = (self.cache / 'samples.csv').read_bytes()
        with patch('attribute_predictor.load_predictor', return_value=(predictor, self.predictor_metadata)):
            output = evaluate_all.main(self.args())
        self.assertEqual(predictor.call_count, 3)
        with (output / 'summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        def find(attribute, metric, comparison='cf_target'):
            return next(row for row in rows if row['family'] == 'effectiveness'
                        and row['comparison'] == comparison and row['attribute'] == attribute
                        and row['metric'] == metric and row['target'] == 'all')
        view = find('view', 'target_accuracy')
        self.assertEqual(float(view['value']), .5)
        self.assertEqual(int(view['n_generated']), 2)
        self.assertEqual(float(find('age', 'target_mae_years')['value']), 1.)
        self.assertEqual(float(find('age', 'target_within_tolerance')['value']), .5)
        self.assertEqual(float(find('age', 'mae_years', 'real_prediction')['value']), 2.5)
        self.assertTrue(any(row['family'] == 'identity' and row['metric'] == 'ssim' for row in rows))
        self.assertEqual(before, (self.cache / 'samples.csv').read_bytes())
        self.assertEqual(json.loads((output / 'config.json').read_text())['status'], 'complete')
        self.assertTrue((output / 'effectiveness/per_sample.csv').is_file())

    def test_predictor_test_split_must_match(self):
        self.predictor_metadata['split_csv_sha256']['test'] = 'd' * 64
        with patch('attribute_predictor.load_predictor', return_value=(Mock(), self.predictor_metadata)):
            with self.assertRaisesRegex(ValueError, 'test-split hash'):
                evaluate_all.main(self.args())
        self.assertFalse((self.root / 'results').exists())

    def test_uniform_cache_includes_same_target_controls(self):
        with (self.cache / 'samples.csv').open() as handle:
            records = list(csv.DictReader(handle))
        for row in records:
            if row['task'] == 'counterfactual' and row['attribute'] == 'view' and row['index'] == '0':
                row['target'] = row['view'] = row['source']
        write_records(self.cache, records)
        self.config.update(counterfactual_target_policy='uniform_v1',
                           counterfactual_targets_sha256=counterfactual_targets_hash(records))
        save_json(self.cache / 'config.json', self.config)
        predictions = [
            dict(view=torch.tensor([[2., 0.], [0., 2.]]), age=torch.tensor([40., 60.])),
            dict(view=torch.tensor([[2., 0.], [2., 0.]]), age=torch.tensor([40., 60.])),
            dict(view=torch.tensor([[2., 0.], [0., 2.]]), age=torch.tensor([50., 70.])),
        ]
        with patch('attribute_predictor.load_predictor',
                   return_value=(Mock(side_effect=predictions), self.predictor_metadata)):
            output = evaluate_all.main(self.args())
        with (output / 'effectiveness/summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        score = next(row for row in rows if row['comparison'] == 'cf_target'
                     and row['attribute'] == 'view' and row['target'] == 'all'
                     and row['metric'] == 'target_accuracy')
        self.assertEqual((float(score['value']), int(score['n_generated'])), (1., 2))
        self.assertEqual(output.name, 'test_n2_seed1_steps50_ema')
        metadata = json.loads((output / 'config.json').read_text())
        self.assertEqual(metadata['counterfactual_targets_sha256'], self.config['counterfactual_targets_sha256'])

    def test_uniform_cache_rejects_changed_target_assignment(self):
        self.config.update(counterfactual_target_policy='uniform_v1', counterfactual_targets_sha256='f' * 64)
        save_json(self.cache / 'config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'target assignments'):
            evaluate_all.main(self.args())

    def test_missing_predictor_fails_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, 'requires --attribute-predictor'):
            evaluate_all.main(self.args('--attribute-predictor', ''))

    def test_existing_metrics_work_without_predictor(self):
        args = self.args('--evaluations', 'identity', '--attribute-predictor', '')
        with patch('attribute_predictor.load_predictor') as load:
            output = evaluate_all.main(args)
        load.assert_not_called()
        self.assertFalse((output / 'effectiveness').exists())


if __name__ == '__main__':
    unittest.main()
