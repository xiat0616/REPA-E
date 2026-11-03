"""Synthetic correctness tests; no data, model training or GPU is needed."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.effectiveness_metrics import EffectivenessMetrics


def metric(scorer, comparison, attribute, name, target='all'):
    return next(row for row in scorer.summaries()
                if (row['comparison'], row['attribute'], row['metric'], row['target'])
                == (comparison, attribute, name, target))


class EffectivenessTests(unittest.TestCase):
    def test_explicit_target_and_no_factual_correctness_filter(self):
        scorer = EffectivenessMetrics(['view'])
        # Both CFs achieve target 1. The second factual image is already
        # misclassified as target 1; it must still contribute to effectiveness.
        rows = [{'attribute': 'view', 'source': '0.0', 'target': '1.0', 'view': '1'}] * 2
        scorer.update_counterfactual('view', {'view': [[0, 5], [0, 5]]},
                                     {'view': [[5, 0], [0, 5]]}, rows, [1, 2])
        scored = metric(scorer, 'cf_target', 'view', 'target_accuracy')
        self.assertEqual((scored['value'], scored['n_generated']), (1.0, 2))
        self.assertEqual(metric(scorer, 'input_target', 'view', 'target_accuracy')['value'], .5)
        self.assertEqual(len(scorer.per_sample), 4)

    def test_source_predictions_fail_requested_target(self):
        scorer = EffectivenessMetrics(['view'])
        scorer.update_counterfactual('view', {'view': [[4, 0]]}, {'view': [[4, 0]]},
                                     [{'source': '0', 'target': '1', 'view': '1'}], [0])
        self.assertEqual(metric(scorer, 'cf_target', 'view', 'target_accuracy')['value'], 0)

    def test_balanced_target_accuracy_uses_represented_classes(self):
        scorer = EffectivenessMetrics(['density'])
        targets = [0, 0, 0, 2]
        predictions = {'density': [[4, 0, 0, 0]] * 4}
        rows = [{'source': '1', 'target': str(target), 'density': str(target)} for target in targets]
        scorer.update_counterfactual('density', predictions, predictions, rows, [0, 1, 2, 3])
        self.assertEqual(metric(scorer, 'cf_target', 'density', 'target_accuracy')['value'], .75)
        balanced = metric(scorer, 'cf_target', 'density', 'balanced_target_accuracy')
        self.assertEqual((balanced['value'], balanced['n_target_groups']), (.5, 2))
        self.assertEqual(metric(scorer, 'cf_target', 'density', 'target_accuracy', '2')['value'], 0)

    def test_age_years_tolerance_inclusive_and_predictions_unclamped(self):
        scorer = EffectivenessMetrics(['age'], age_tolerance_years=5)
        rows = [{'source': '50', 'target': '60', 'age': '60'},
                {'source': '90', 'target': '100', 'age': '100'}]
        scorer.update_counterfactual('age', {'age': [55.0, 110.0]}, {'age': [50., 90.]}, rows, [0, 1])
        self.assertEqual(metric(scorer, 'cf_target', 'age', 'target_mae_years')['value'], 7.5)
        self.assertAlmostEqual(metric(scorer, 'cf_target', 'age', 'target_rmse_years')['value'], math.sqrt(62.5))
        self.assertEqual(metric(scorer, 'cf_target', 'age', 'target_within_tolerance')['value'], .5)
        self.assertEqual(scorer.per_sample[2]['prediction'], 110.)

    def test_real_diagnostics_and_absent_interventions(self):
        scorer = EffectivenessMetrics(['view', 'age'])
        scorer.update_real({'view': [[0, 2], [2, 0]], 'age': [40., 65.]},
                           [{'view': '1', 'age': '40'}, {'view': '1', 'age': '60'}], [0, 1])
        self.assertEqual(metric(scorer, 'real_prediction', 'view', 'accuracy')['value'], .5)
        self.assertEqual(metric(scorer, 'real_prediction', 'age', 'mae_years')['value'], 2.5)
        for row in scorer.summaries():
            if row['comparison'] != 'real_prediction':
                self.assertEqual((row['value'], row['n_generated'], row['status']),
                                 (None, 0, 'no_interventions'))

    def test_reject_bad_tolerance_and_schema(self):
        for tolerance in [0, -1, float('nan'), float('inf')]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                EffectivenessMetrics(['age'], tolerance)
        with self.assertRaises(ValueError):
            EffectivenessMetrics(['unknown'])

    def test_reject_nonfinite_predictions_wrong_shapes_and_labels(self):
        for predictions in [{'view': [[float('nan'), 0]]}, {'view': [[1, 2, 3]]}, {'view': [1]}, {'view': []}]:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                EffectivenessMetrics(['view']).update_real(predictions, [{'view': '0'}], [0])
        for prediction in [float('nan'), float('inf'), [50.]]:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                EffectivenessMetrics(['age']).update_real({'age': [prediction]}, [{'age': '50'}], [0])
        for label in ['-1', '2', '0.5']:
            with self.subTest(label=label), self.assertRaises(ValueError):
                EffectivenessMetrics(['view']).update_real({'view': [[1, 0]]}, [{'view': label}], [0])

    def test_reject_inconsistent_noop_or_duplicate_records(self):
        predictions = {'view': [[1, 0]]}
        for row in [{'source': '1', 'target': '1'}, {'source': '0', 'target': '1', 'view': '0'},
                    {'source': '0', 'target': '1', 'attribute': 'density'}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                EffectivenessMetrics(['view']).update_counterfactual('view', predictions, predictions, [row], [0])
        scorer = EffectivenessMetrics(['view'])
        scorer.update_real(predictions, [{'view': '0'}], [0])
        with self.assertRaises(ValueError):
            scorer.update_real(predictions, [{'view': '0'}], [0])
        self.assertEqual(len(scorer.per_sample), 1)

    def test_invalid_batch_does_not_partially_update(self):
        scorer = EffectivenessMetrics(['view', 'age'])
        with self.assertRaises(ValueError):
            scorer.update_real({'view': [[2, 0]], 'age': [float('inf')]},
                               [{'view': '0', 'age': '50'}], [0])
        self.assertEqual(scorer.per_sample, [])
        # A finite but numerically unusable prediction must also fail cleanly.
        with self.assertRaises(ValueError):
            scorer.update_real({'view': [[2, 0]], 'age': [1e308]},
                               [{'view': '0', 'age': '50'}], [0])
        self.assertEqual(scorer.per_sample, [])


if __name__ == '__main__':
    unittest.main()
