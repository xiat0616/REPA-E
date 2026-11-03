"""Attribute-predictor checks of requested counterfactual interventions.

Categorical effectiveness is argmax accuracy against the *requested target*,
and balanced target accuracy averages those recalls over represented target
categories. Age is continuous: MAE/RMSE use years, and the optional success
rate uses an explicit absolute-error tolerance in years. No score combines
categorical accuracy with age error. Every eligible counterfactual is included,
including images whose factual attribute the predictor gets wrong.

``input_target`` applies the same target checks to the corresponding input
images as a no-edit diagnostic. ``real_prediction`` checks real-image labels;
it is a cache diagnostic, not an independent held-out predictor qualification.
Predictions can be tensors, ndarrays or ordinary Python lists. Accumulation
uses only the standard library and never changes the predictor or images.
"""
from __future__ import annotations

import math
import statistics


CLASS_COUNTS = {'view': 2, 'density': 4, 'scanner': 5}
ATTRIBUTES = (*CLASS_COUNTS, 'age')


def _values(values):
    if hasattr(values, 'detach'):
        values = values.detach().cpu()
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise ValueError('Predictions must have an explicit batch dimension.')
    return values


def _finite(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{description} must be numeric and finite.') from error
    if not math.isfinite(number):
        raise ValueError(f'{description} must be numeric and finite.')
    return number


def _label(value, attribute):
    number = _finite(value, f'{attribute} label')
    if attribute == 'age':
        if not 0 <= number <= 100:
            raise ValueError('Age labels must be in [0,100] years.')
        return number
    if number != int(number) or not 0 <= number < CLASS_COUNTS[attribute]:
        raise ValueError(f'{attribute} labels must be integer class IDs in [0,{CLASS_COUNTS[attribute] - 1}].')
    return int(number)


def _predictions(predictions, attribute, size):
    if attribute not in predictions:
        raise ValueError(f'Predictor output is missing {attribute}.')
    values = _values(predictions[attribute])
    if len(values) != size:
        raise ValueError(f'{attribute} prediction batch size differs from the records.')
    if attribute == 'age':
        # B rather than Bx1 is intentional: age is already expressed in years.
        return [_finite(value, 'Age prediction in years') for value in values]
    result = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != CLASS_COUNTS[attribute]:
            raise ValueError(f'{attribute} logits must have shape Bx{CLASS_COUNTS[attribute]}.')
        logits = [_finite(value, f'{attribute} logit') for value in row]
        result.append(max(range(len(logits)), key=logits.__getitem__))
    return result


def _indices(indices, size):
    values = _values(indices)
    if len(values) != size:
        raise ValueError('Index batch size differs from the records.')
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError('Indices must be nonnegative integers.')
    return values


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else None


class EffectivenessMetrics:
    """Collect factual diagnostics and paired requested-target effectiveness."""

    def __init__(self, attributes, age_tolerance_years=5.0):
        self.attributes = list(dict.fromkeys(attributes))
        if not self.attributes or not set(self.attributes).issubset(ATTRIBUTES):
            raise ValueError(f'Attributes must be a nonempty subset of {ATTRIBUTES}.')
        self.age_tolerance_years = _finite(age_tolerance_years, 'Age tolerance in years')
        if self.age_tolerance_years <= 0:
            raise ValueError('Age tolerance must be positive.')
        self.per_sample = []
        self._seen = set()

    def _row(self, index, comparison, attribute, source, target, prediction):
        error = abs(prediction - target) if attribute == 'age' else None
        squared_error = _finite(error * error, 'Squared age error') if error is not None else ''
        return {
            'index': index, 'comparison': comparison, 'attribute': attribute,
            'source': source, 'target': target, 'prediction': prediction,
            'success': int(error <= self.age_tolerance_years) if error is not None else int(prediction == target),
            'absolute_error_years': error if error is not None else '',
            'squared_error_years': squared_error,
        }

    def update_real(self, predictions, factual_rows, indices):
        """Compare real-image predictions with the saved factual labels."""
        indices = _indices(indices, len(factual_rows))
        prepared = []
        for attribute in self.attributes:
            outputs = _predictions(predictions, attribute, len(factual_rows))
            labels = [_label(row[attribute], attribute) for row in factual_rows]
            prepared.extend((index, 'real_prediction', attribute, '', label, prediction)
                            for index, label, prediction in zip(indices, labels, outputs))
        self._extend(prepared)

    def update_counterfactual(self, attribute, predictions, factual_predictions, cf_rows, indices):
        """Score CFs and their inputs against explicit intervention targets."""
        if attribute not in self.attributes:
            raise ValueError(f'Unrequested effectiveness attribute: {attribute}.')
        indices = _indices(indices, len(cf_rows))
        outputs = _predictions(predictions, attribute, len(cf_rows))
        inputs = _predictions(factual_predictions, attribute, len(cf_rows))
        prepared = []
        for index, row, prediction, factual_prediction in zip(indices, cf_rows, outputs, inputs):
            if row.get('attribute', attribute) != attribute:
                raise ValueError('Counterfactual manifest attribute differs from the requested scorer.')
            source = _label(row['source'], attribute)
            target = _label(row['target'], attribute)
            if math.isclose(source, target, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Effectiveness requires a genuine source-to-target intervention.')
            if attribute in row and not math.isclose(_label(row[attribute], attribute), target,
                                                     rel_tol=0, abs_tol=1e-4):
                raise ValueError('Manifest attribute label disagrees with the requested target.')
            prepared.extend(((index, 'cf_target', attribute, source, target, prediction),
                             (index, 'input_target', attribute, source, target, factual_prediction)))
        self._extend(prepared)

    def _extend(self, prepared):
        # Reject malformed/duplicate batches before adding partial results.
        keys = [(row[0], row[1], row[2]) for row in prepared]
        if len(set(keys)) != len(keys) or any(key in self._seen for key in keys):
            raise ValueError('Duplicate effectiveness observations in a batch or previous update.')
        rows = [self._row(*row) for row in prepared]
        self.per_sample.extend(rows)
        self._seen.update(keys)

    def _summary(self, comparison, attribute, target, group):
        real = comparison == 'real_prediction'
        count = len(group)
        status = 'ok' if count else ('no_predictions' if real else 'no_interventions')
        targets = {row['target'] for row in group}
        base = dict(comparison=comparison, attribute=attribute, target=target,
                    n_real=count, n_generated=count, status=status,
                    n_target_groups=len(targets))

        def result(metric, value, std=None):
            return dict(base, metric=metric, value=value, std=std)

        if attribute == 'age':
            prefix = '' if real else 'target_'
            errors = [row['absolute_error_years'] for row in group]
            squares = [row['squared_error_years'] for row in group]
            rows = [result(prefix + 'mae_years', statistics.mean(errors) if count else None,
                           _std(errors)),
                    result(prefix + 'rmse_years', math.sqrt(statistics.mean(squares)) if count else None)]
            if not real:
                success = [row['success'] for row in group]
                rows.append(result('target_within_tolerance', statistics.mean(success) if count else None,
                                   _std(success)))
            return rows
        success = [row['success'] for row in group]
        rows = [result('accuracy' if real else 'target_accuracy', statistics.mean(success) if count else None,
                       _std(success))]
        if target == 'all':
            recalls = [statistics.mean(row['success'] for row in group if row['target'] == label)
                       for label in sorted(targets)]
            rows.append(result('balanced_accuracy' if real else 'balanced_target_accuracy',
                               statistics.mean(recalls) if count else None))
        return rows

    def summaries(self):
        """Return macro and target-group summaries; unavailable scores are None.

        Balanced accuracy includes only represented target classes; the number
        of represented classes is explicit in ``n_target_groups``. Standard
        deviations describe per-image variation, not uncertainty intervals;
        aggregate balanced accuracy and RMSE have no per-image standard deviation.
        """
        groups = {}
        for row in self.per_sample:
            groups.setdefault((row['comparison'], row['attribute']), []).append(row)
        summaries = []
        for comparison in ('real_prediction', 'cf_target', 'input_target'):
            for attribute in self.attributes:
                group = groups.get((comparison, attribute), [])
                summaries.extend(self._summary(comparison, attribute, 'all', group))
                if comparison != 'real_prediction' and attribute != 'age':
                    for target in sorted({row['target'] for row in group}):
                        selected = [row for row in group if row['target'] == target]
                        summaries.extend(self._summary(comparison, attribute, str(target), selected))
        return summaries
