"""Numerical tests of physical frequency placement, filtering and crop phase."""
import unittest
import numpy as np
from scipy.signal import freqz
from native import BANDS, FS, GUARD, TAPS, kernel, transform, oscillator, crop_start, WINDOW, LENGTH


class NativeTests(unittest.TestCase):
    def test_tone_placement_phase_and_delay(self):
        for spec in BANDS.values():
            for center in spec['admitted_centers']:
                shift = center - spec['center_hz']
                for target_frequency in (-11_000_000, 3_000_000, 13_000_000):
                    origin = 9_000_000_137
                    x = oscillator(32768, target_frequency - shift, origin)
                    actual = transform(x, center, origin)
                    expected = oscillator(len(actual), target_frequency, origin + GUARD)
                    self.assertLess(np.mean(abs(actual - expected)**2), 1e-8)

    def test_prefilter_rejects_would_be_alias(self):
        center = 5760000000  # shift -20 MHz; -40 MHz would alias to +40 MHz.
        x = oscillator(32768, -40000000, 131)
        self.assertLess(np.mean(abs(transform(x, center, 131))**2), 1e-8)

    def test_chunk_origin_and_guard_consistency(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=98304) + 1j*rng.normal(size=98304)
        full = transform(x, 5800000000, 1097)
        a, b = 16384, 81920
        part = transform(x[a:b], 5800000000, 1097 + a)
        expected = full[a:b-2*GUARD]
        self.assertLess(np.mean(abs(part-expected)**2)/np.mean(abs(expected)**2), 1e-12)

    def test_linearity_and_no_periodic_edge_wrap(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=32768) + 1j*rng.normal(size=32768)
        y = rng.normal(size=32768) + 1j*rng.normal(size=32768)
        a, b = .31+.2j, -.18+.7j
        lhs = transform(a*x+b*y, 2450000000, 307)
        rhs = a*transform(x, 2450000000, 307)+b*transform(y, 2450000000, 307)
        self.assertLess(np.mean(abs(lhs-rhs)**2)/np.mean(abs(lhs)**2), 1e-12)
        impulse = np.zeros(32768, complex); impulse[0] = 1
        self.assertLess(np.max(abs(transform(impulse, 2450000000))), 1e-12)

    def test_filter_pass_and_stop_bands(self):
        for spec in BANDS.values():
            for center in spec['admitted_centers']:
                freq, response = freqz(kernel(center), worN=131072, whole=True, fs=FS)
                freq = (freq+FS/2) % FS-FS/2
                common = freq+center-spec['center_hz']
                inside = abs(common) <= spec['pass_half_hz']
                outside = abs(common) >= spec['stop_half_hz']
                self.assertLess(np.max(abs(abs(response[inside])-1)), 1e-4)
                self.assertLess(np.max(abs(response[outside])), 1e-4)

    def test_crops_stay_inside_valid_context(self):
        for start in (0, 1, 4095, 4096, 1800000, 2097152-WINDOW):
            mapped = crop_start(start)
            self.assertGreaterEqual(mapped, 0)
            self.assertLessEqual(mapped+WINDOW, LENGTH)


if __name__ == '__main__':
    unittest.main()
