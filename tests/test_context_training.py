import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from drone_rf.context_training_data import ContextMixtures, make_schedule, write_json
from drone_rf.data import sha256
from drone_rf.waveform import analyze, synthesize, complex_si_sdr, waveform_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_mixture_schedule import DTYPE as BASE_DTYPE
from train_context_experiment import objective, serialize_extended
from drone_rf.context_model import ContextualSeparator


class ContextTrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_count_balanced_repeatable_group_split_schedules(self):
        clips = [dict(role=role, category=cat, samples=2097152, fs_hz=100000000)
                 for role in ('train_pack', 'validation_pack') for cat in ('A', 'B', 'C', 'D', 'E')]
        train = make_schedule(clips, 'train_pack', epochs=2, examples_per_count=800)
        np.testing.assert_array_equal(train, make_schedule(clips, 'train_pack', epochs=2, examples_per_count=800))
        for epoch in (1, 2):
            rows = train[train['epoch'] == epoch]
            self.assertEqual([int(np.sum(rows['count'] == n)) for n in (1, 2, 3)], [800] * 3)
        val = make_schedule(clips, 'validation_pack', epochs=1, examples_per_count=210)
        for data, role in ((train, 'train_pack'), (val, 'validation_pack')):
            for row in data:
                selected = row['indices'][:row['count']]
                self.assertTrue(all(clips[int(i)]['role'] == role for i in selected))
                self.assertEqual(len(set(selected)), row['count'])
                self.assertTrue(np.all(row['indices'][row['count']:] == -1))
        self.assertFalse(np.array_equal(train[:2400], train[2400:]))

    def test_short_loader_replays_full_context_and_rejects_changed_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clips = []
            for i, category in enumerate(('DJI AVATA2', 'DJI MINI3', 'DJI MINI4 PRO')):
                signal = np.ones(2097152, np.complex64) * (i + 1j)
                signal[120:376] *= .1
                path = root / f'clip{i}.npy'
                np.save(path, signal)
                clips.append(dict(clip_id=str(i), category=category, role='train_pack',
                    pack_id=f'{i}.xml', fs_hz=100000000, samples=len(signal), center_hz=5800000000,
                    mean_power=float(np.mean(np.abs(signal.astype(np.complex128)) ** 2)),
                    cache_path=str(path), cache_sha256=sha256(path)))
            manifest = root / 'manifest.json'
            write_json(manifest, dict(clips=clips, exact_cross_role_duplicate_groups=0))
            write_json(root / 'CLIP_INDEX.json', dict(clips=[{k: v for k, v in c.items() if k != 'cache_path'} for c in clips]))
            np.save(root / 'TRAIN_SCHEDULE.npy', np.array([(1, 0, 1, 0, 0, 0, 0)], dtype=BASE_DTYPE))
            write_json(root / 'COMPLETE.json', dict(status='CPU_CACHE_REPLAY_AND_MIXTURE_SCHEDULE_COMPLETE',
                source_manifest_sha256=sha256(manifest), window_samples=256,
                files={n: sha256(root / n) for n in ('CLIP_INDEX.json', 'TRAIN_SCHEDULE.npy')}))
            rows = make_schedule(clips, 'train_pack', epochs=1, examples_per_count=1, length=256)
            rows['crop_start'] = 120
            np.save(root / 'TRAIN.npy', rows)
            write_json(root / 'PREPARATION.json', dict(status='PINNED_SYNTHETIC_COMPONENT_EXPERIMENT',
                files={'TRAIN.npy': sha256(root / 'TRAIN.npy')}, manifest=str(manifest),
                manifest_sha256=sha256(manifest), base_schedule=str(root), window_samples=256,
                context_samples=2097152, fs_hz=100000000))
            dataset = ContextMixtures(root, 'train_pack', use_features=False)
            examples = [dataset.full_example(i) for i in range(3)]
            (root / 'features').mkdir()
            np.save(dataset.cache_stem.with_suffix('.npy'), np.stack([x['context_features'] for x in examples]))
            write_json(dataset.cache_stem.with_suffix('.json'), dict(status='FEATURES_READY',
                rows_sha256=dataset.rows_hash, feature_sha256=sha256(dataset.cache_stem.with_suffix('.npy'))))
            dataset.load_features()
            for i, example in enumerate(examples):
                short = dataset[i]
                for key in ('mixture', 'references', 'active', 'context_features'):
                    np.testing.assert_array_equal(short[key], example[key])
                self.assertFalse(short['physical_count_eligible'])
            with dataset.cache_stem.with_suffix('.npy').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Feature cache changed'):
                dataset.load_features()

    def test_complex_stft_round_trip_and_sum_survive_synthesis(self):
        x = torch.randn(2, 4096, dtype=torch.complex64)
        spectrum = analyze(x)
        restored = synthesize(spectrum, x.shape[-1])
        torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)
        slots = spectrum[:, None].expand(-1, 4, -1, -1) / 4
        waves = synthesize(slots, x.shape[-1])
        torch.testing.assert_close(waves.sum(1), x, atol=2e-6, rtol=2e-6)

    def test_waveform_metrics_swap_and_complex_scale_convention(self):
        refs = torch.randn(1, 3, 1000, dtype=torch.complex64)
        estimates = torch.cat([refs[:, [2, 0, 1]], torch.zeros_like(refs[:, :1])], dim=1)
        active = torch.ones(1, 3, dtype=torch.bool)
        metrics = waveform_metrics(estimates, refs, active, refs.sum(1))
        self.assertEqual(metrics['nmse'].max().item(), 0)
        noisy = refs + .1 * torch.randn_like(refs)
        torch.testing.assert_close(complex_si_sdr(noisy, refs), complex_si_sdr(noisy * (1.3 + .8j), refs), atol=2e-5, rtol=1e-5)

    def test_collapsed_output_is_not_reported_as_finite_zero_db(self):
        reference = torch.randn(2, 257, dtype=torch.complex64)
        score = complex_si_sdr(torch.zeros_like(reference), reference)
        self.assertTrue(torch.isnan(score).all())
        values, statuses = serialize_extended(score)
        self.assertEqual(values, [None, None])
        self.assertEqual(statuses, ['undefined', 'undefined'])
        json.dumps(dict(values=values, statuses=statuses), allow_nan=False)

    def test_full_width_waveform_loss_path_has_finite_gradients(self):
        torch.manual_seed(0)
        net = ContextualSeparator('tcn')
        refs = torch.randn(1, 3, 4096, dtype=torch.complex64)
        item = dict(mixture=refs.sum(1), references=refs,
                    active=torch.ones(1, 3, dtype=torch.bool),
                    context_features=torch.randn(1, 65, 256),
                    crop_start=torch.tensor([200000]), construction_count=torch.tensor([3]))
        loss, wave, count = objective(net, item, {'construction_count_loss_weight': .1})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (net.output.weight, net.context_projection.weight, net.count_head[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
