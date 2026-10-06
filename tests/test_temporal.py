import unittest
import numpy as np
from drone_rf.temporal import lag_similarity, local_peak, block_shuffle_rank, power_features


class TemporalChecks(unittest.TestCase):
    def test_fft_agrees_with_direct_multifeature_overlap(self):
        x = np.random.default_rng(0).normal(size=(3, 100))
        x -= x.mean(1, keepdims=True)
        got = lag_similarity(x, 30)
        for k in range(31):
            a, b = x[:, :100-k], x[:, k:]
            expected = np.sum(a*b) / np.sqrt(np.sum(a*a)*np.sum(b*b))
            self.assertAlmostEqual(got[k], expected, places=12)

    def test_periodic_bursts_survive_but_block_shuffle_breaks_recurrence(self):
        x = ((np.arange(2048) % 160) < 32).astype(float)
        curve = lag_similarity(x, 513)
        peak = local_peak(curve, 128, 512)
        self.assertEqual(peak['lag_frames'], 160)
        self.assertGreater(peak['correlation'], .999)
        rank = block_shuffle_rank(x, peak['correlation'], 128, 512, seed=0)
        self.assertLess(rank['exceedance_fraction'], .05)

    def test_constant_is_not_periodic_and_monotonic_is_not_a_peak(self):
        self.assertTrue(np.all(lag_similarity(np.ones(2048), 512) == 0))
        self.assertIsNone(local_peak(lag_similarity(np.ones(2048), 513), 128, 512))
        self.assertIsNone(local_peak(np.linspace(1, 0, 514), 128, 512))

    def test_frequency_profile_preserves_hops_without_energy_change(self):
        t = np.arange(1024)
        tones = np.concatenate([np.exp(2j*np.pi*k*t/1024) for k in (48, 240, 48, 240)])
        envelope, profile = power_features(tones)
        np.testing.assert_allclose(envelope, 1.)
        np.testing.assert_allclose(profile.sum(0), 1.)
        peaks = profile.argmax(0)
        self.assertEqual(peaks[0], peaks[2]); self.assertEqual(peaks[1], peaks[3])
        self.assertNotEqual(peaks[0], peaks[1])


if __name__ == '__main__':
    unittest.main()
