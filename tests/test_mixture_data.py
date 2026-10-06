import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from drone_rf.data import ScheduledMixtures, long_context_mixture, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_mixture_schedule import DTYPE


class MixtureDataChecks(unittest.TestCase):
    def test_long_power_sir_and_phase_have_known_solution(self):
        first = np.full(32, 2 + 0j)
        second = np.full(32, 0 + 3j)
        mixture, sources = long_context_mixture(first, second, [4, 9], 10, [np.pi / 2, 0])
        np.testing.assert_allclose(sources[0], 1j * np.sqrt(10 / 11), atol=1e-7)
        np.testing.assert_allclose(sources[1], 1j / np.sqrt(11), atol=1e-7)
        np.testing.assert_array_equal(mixture, sources.sum(0))

    def test_local_activity_not_renormalized_and_silence_preserved(self):
        # Both full contexts have unit power; first window power is only 0.01.
        mixture, sources = long_context_mixture(np.full(32, .1 + 0j), np.ones(32, complex), [1, 1], 0, [0, 0])
        powers = np.mean(np.abs(sources) ** 2, axis=1)
        self.assertAlmostEqual(10 * np.log10(powers[0] / powers[1]), -20, places=5)
        _, silent = long_context_mixture(np.zeros(32, complex), np.ones(32, complex), [1, 1], 0, [0, 0])
        self.assertTrue(np.all(silent[0] == 0))

    def fixture(self, root):
        clips = []
        for i, category in enumerate(('DJI AVATA2', 'DJI MINI3')):
            p = root / f'{i}.npy'
            np.save(p, np.full(128, 1 + i * 1j, np.complex64))
            clips.append(dict(clip_id=str(i), cache_path=str(p), cache_sha256=sha256(p),
                pack_id=f'{i}.xml', category=category, role='train_pack', samples=128,
                mean_power=1 + i, fs_hz=100000000, center_hz=5800000000))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(dict(clips=clips, exact_cross_role_duplicate_groups=0)))
        (root / 'CLIP_INDEX.json').write_text(json.dumps(dict(clips=[
            {k: v for k, v in c.items() if k != 'cache_path'} for c in clips])))
        np.save(root / 'TRAIN_SCHEDULE.npy', np.array([(1, 0, 1, 10, 10, 0, .3)], dtype=DTYPE))
        receipt = dict(status='CPU_CACHE_REPLAY_AND_MIXTURE_SCHEDULE_COMPLETE',
            source_manifest_sha256=sha256(manifest), window_samples=32,
            files={n: sha256(root / n) for n in ['CLIP_INDEX.json', 'TRAIN_SCHEDULE.npy']})
        (root / 'COMPLETE.json').write_text(json.dumps(receipt))
        return manifest

    def test_receipt_integrity_and_changed_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            dataset = ScheduledMixtures(manifest, root, 'train_pack', max_open=1)
            sample = dataset[0]
            np.testing.assert_array_equal(sample['mixture'], sample['references'].sum(0))
            self.assertEqual(sample['references'].shape, (2, 32))
            self.assertEqual(len(dataset._maps), 1)
            np.save(root / '0.npy', np.zeros(128, np.complex64))
            with self.assertRaisesRegex(ValueError, 'changed during use'):
                dataset[0]
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ScheduledMixtures(manifest, root, 'train_pack')[0]
            with (root / 'TRAIN_SCHEDULE.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Pinned schedule'):
                ScheduledMixtures(manifest, root, 'train_pack')

    def test_bad_role_cannot_load_training_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            with self.assertRaises(ValueError):
                ScheduledMixtures(manifest, root, 'heldout_model')


if __name__ == '__main__':
    unittest.main()
