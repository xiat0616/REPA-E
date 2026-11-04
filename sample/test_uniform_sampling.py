"""Sampler integration checks without loading checkpoints, images or GPUs."""
from argparse import Namespace
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample.sample_checkpoint import make_intervention
from sample.sample_io import counterfactual_targets_hash


class UniformSamplingTests(unittest.TestCase):
    def setUp(self):
        self.parents = dict(view=F.one_hot(torch.zeros(128, dtype=torch.long), 2).float(),
                            density=F.one_hot(torch.zeros(128, dtype=torch.long), 4).float(),
                            scanner=F.one_hot(torch.zeros(128, dtype=torch.long), 5).float(),
                            age=torch.full((128, 1), .5))
        self.args = Namespace(seed=1234, target_view=None, target_density=None,
                              target_scanner=None, target_age=None)

    def test_all_uniform_draws_survive_including_source_class(self):
        for attribute in self.parents:
            with self.subTest(attribute=attribute):
                result, eligible = make_intervention(self.parents, attribute, self.args,
                    indices=range(128), age_range_years=(30., 90.))
                self.assertEqual(int(eligible.sum()), 128)
                if attribute != 'age':
                    target = result[attribute].argmax(1)
                    self.assertIn(0, target.tolist())
                    self.assertEqual(len(torch.unique(target)), self.parents[attribute].shape[1])

    def test_explicit_target_overrides_uniform_default_and_keeps_noops(self):
        for attribute, target in [('view', 0), ('density', 3), ('scanner', 0), ('age', 50.)]:
            with self.subTest(attribute=attribute):
                setattr(self.args, 'target_' + attribute, target)
                result, eligible = make_intervention(self.parents, attribute, self.args, indices=range(128))
                self.assertTrue(bool(eligible.all()))
                if attribute == 'age':
                    self.assertTrue(torch.equal(result[attribute], torch.full_like(result[attribute], .5)))
                else:
                    self.assertTrue(bool((result[attribute].argmax(1) == target).all()))
                setattr(self.args, 'target_' + attribute, None)

    def test_assignment_fingerprint_ignores_image_outputs_and_row_order(self):
        rows = [dict(index=2, task='counterfactual', attribute='scanner', source=0., target=0., sha256='a'),
                dict(index=1, task='counterfactual', attribute='age', source=50., target=70., sha256='b')]
        expected = counterfactual_targets_hash(rows)
        copied = [{**row, 'sha256': 'different'} for row in reversed(rows)]
        self.assertEqual(counterfactual_targets_hash(copied), expected)
        copied[0]['target'] = 71.
        self.assertNotEqual(counterfactual_targets_hash(copied), expected)


if __name__ == '__main__':
    unittest.main()
