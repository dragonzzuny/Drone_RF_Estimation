import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_context_evidence import (band_count_baseline, source_power_diagnostics,
                                    summarize_rows, validate_rows, schedule_metadata)
from drone_rf.context_training_data import DTYPE
from drone_rf.waveform import predict


def metric_row(count, error):
    return dict(index=count - 1, construction_count=count,
        predicted_construction_count=count, active=[i < count for i in range(3)],
        nmse=[error] * count, si_sdr=[0.] * count, si_sdr_improvement=[0.] * count,
        reference_power=[1 / count] * count, all_components_positive_improvement=False,
        inactive_leak=0., background_nmse=0.)


class EvidenceChecks(unittest.TestCase):
    def test_single_source_improvement_does_not_hide_unchanged_separation(self):
        before = [metric_row(1, 9.), metric_row(2, 1.), metric_row(3, 2.)]
        after = [metric_row(1, 0.), metric_row(2, 1.), metric_row(3, 2.)]
        a, b = summarize_rows(before), summarize_rows(after)
        self.assertLess(b['original_selection_macro_nmse'], a['original_selection_macro_nmse'])
        self.assertEqual(a['separation_only_count_macro_nmse'], 1.5)
        self.assertEqual(b['separation_only_count_macro_nmse'], 1.5)
        self.assertFalse(b['selection_changed'])

    def test_band_baseline_is_fitted_without_validation_labels(self):
        train = [dict(band='low', count=c) for c in [1, 1, 2]]
        train += [dict(band='high', count=c) for c in [1, 2, 3, 3]]
        validation = [dict(band='low', count=2)] * 20 + [dict(band='high', count=1)] * 20
        result = band_count_baseline(train, validation)
        self.assertEqual(result['prediction_by_band'], {'low': 1, 'high': 3})
        self.assertEqual(result['validation_accuracy'], 0.)

    def test_local_power_keeps_quiet_windows_and_strength_reversal(self):
        # First component is nominally stronger, but 100x quieter than its own
        # context average in this crop. The local power direction is reversed.
        d = source_power_diagnostics([.01 * 10 / 11, 1 / 11], [10, 0])
        np.testing.assert_allclose(d['local_relative_long_db'], [-20, 0], atol=1e-12)
        np.testing.assert_allclose(d['nominal_sir_db'], [10, -10], atol=1e-12)
        np.testing.assert_allclose(d['local_sir_db'], [-10, 10], atol=1e-12)
        triple = source_power_diagnostics([1 / 3] * 3, [0] * 3)
        np.testing.assert_allclose(triple['local_sir_db'], [-10 * np.log10(2)] * 3)
        self.assertIsNone(source_power_diagnostics([1], [0])['local_sir_db'])
        with self.assertRaises(ValueError):
            source_power_diagnostics([1, 0], [0, 0])

    def test_validation_integrity_rejects_wrong_epoch_duplicate_rows_and_aggregate(self):
        schedule = np.zeros(3, dtype=DTYPE)
        schedule['count'] = [1, 2, 3]
        doc = dict(epoch=1, rows=[metric_row(c, .5) for c in (1, 2, 3)], macro_component_nmse=.5)
        self.assertEqual(len(validate_rows(doc, schedule, 1)[0]), 3)
        with self.assertRaisesRegex(ValueError, 'epoch'):
            validate_rows(doc, schedule, 2)
        changed = copy.deepcopy(doc)
        changed['rows'][2]['index'] = 1
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows(changed, schedule, 1)
        changed = copy.deepcopy(doc)
        changed['macro_component_nmse'] = .1
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            validate_rows(changed, schedule, 1)

    def test_same_band_admission_rejects_cross_band_or_heldout(self):
        clips = [dict(category=c, role='validation_pack', center_hz=f, pack_id=c, clip_id=c)
                 for c, f in [('DJI AVATA2', 5.8e9), ('DJI MINI3', 2.45e9)]]
        schedule = np.zeros(1, dtype=DTYPE)
        schedule['count'] = 2
        schedule['indices'] = [0, 1, -1]
        with self.assertRaisesRegex(ValueError, 'Cross-band'):
            schedule_metadata(schedule, clips, 'validation_pack')
        clips[1]['center_hz'] = 5.8e9
        self.assertEqual(schedule_metadata(schedule, clips, 'validation_pack')[0]['count'], 2)
        clips[1]['role'] = 'heldout'
        with self.assertRaisesRegex(ValueError, 'role'):
            schedule_metadata(schedule, clips, 'validation_pack')

    def test_inference_never_reads_reference_count_or_identity(self):
        # Exercise the actual STFT/inverse-STFT inference wrapper, with a model
        # probe instead of a second GPU job. Any extra batch access is an error.
        torch.set_num_threads(1)
        class ObservedOnly(dict):
            def __getitem__(self, key):
                if key not in {'mixture', 'context_features', 'crop_start'}:
                    raise AssertionError('Reference information read during prediction')
                return super().__getitem__(key)
        class Probe:
            def __call__(self, z, context, start):
                return dict(estimates=z[:, None].expand(-1, 4, -1, -1) / 4,
                            count_logits=torch.zeros(z.shape[0], 3))
        mix = torch.randn(1, 2048, dtype=torch.complex64)
        batch = ObservedOnly(mixture=mix, context_features=torch.randn(1, 65, 256),
                             crop_start=torch.tensor([0]))
        predictions, _ = predict(Probe(), batch)
        torch.testing.assert_close(predictions.sum(1), mix, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
