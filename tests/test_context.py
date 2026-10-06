import unittest

import numpy as np
import torch

from drone_rf.context_data import contextual_mixture, mixture_context_features
from drone_rf.context_model import ContextualSeparator, align_context
from drone_rf.losses import pit_waveform_loss, source_count_loss
from drone_rf.model import ComplexSeparator


class ContextChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_one_two_three_mixture_crop_and_features_are_observation_only(self):
        rng = np.random.default_rng(9)
        records = [(rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
                   for _ in range(3)]
        # Independent replay of gains and *complex* mixing (including cross terms).
        for count in (1, 2, 3):
            powers = np.array([float(np.mean(np.abs(x.astype(np.complex128)) ** 2)) for x in records[:count]])
            levels = np.array([-10., 0., 10.][:count])
            phases = np.array([.1, .2, .3][:count])
            sample = contextual_mixture(records[:count], powers, levels, phases, 101, 257,
                                        fft_size=64, bands=8, pool_frames=4)
            weights = 10 ** ((levels - levels.max()) / 10)
            gains = np.sqrt(weights / weights.sum() / powers) * np.exp(1j * phases)
            components = [(x * g).astype(np.complex64) for x, g in zip(records, gains)]
            expected = np.zeros(4096, np.complex64)
            for component in components:
                expected += component
            np.testing.assert_array_equal(sample['mixture'], expected[101:358])
            np.testing.assert_array_equal(sample['references'].sum(0), sample['mixture'])
            observed = mixture_context_features(expected, 64, 8, 4)
            np.testing.assert_array_equal(sample['context_features'], observed['context_features'])
            self.assertFalse(sample['references'][count:].any())
            self.assertFalse(sample['count_eligible'])
            self.assertEqual(sample['construction_count'], count)

    def test_destructive_interference_is_not_sum_of_source_spectrograms(self):
        x = np.ones(2048, np.complex64)
        sample = contextual_mixture([x, -x], [1, 1], [0, 0], [0, 0], 0, 100,
                                    fft_size=64, bands=8, pool_frames=4)
        self.assertFalse(sample['mixture'].any())
        self.assertFalse(sample['context_features'].any())
        self.assertTrue(sample['references'].any())
        self.assertFalse(sample['count_eligible'])

    def test_alignment_uses_sample_centers_not_resized_time_axes(self):
        # Tokens encode their sample coordinates. Interpolation should reproduce
        # exact short-window pooling centers, including an odd fine frame count.
        positions = 4095.5 + 8192 * torch.arange(256)
        encoded = positions[None, None].expand(2, 2, -1)
        starts = torch.tensor([20000, 80000])
        aligned = align_context(encoded, starts, fine_frames=500)
        expected = starts[:, None] + (16 * torch.arange(31)[None] + 7.5) * 128
        torch.testing.assert_close(aligned[:, 0], expected, atol=.2, rtol=0)
        edge = align_context(encoded[:1], torch.tensor([0]), fine_frames=16)
        self.assertAlmostEqual(edge[0, 0, 0].item(), 4095.5, places=2)

    def test_three_slot_pit_handles_every_requested_count(self):
        for count in (1, 2, 3):
            refs = torch.randn(1, 3, 129, dtype=torch.complex64)
            refs[:, count:] = 0
            active = torch.arange(3)[None] < count
            bg = .01 * torch.randn(1, 129, dtype=torch.complex64)
            estimates = torch.cat([refs[:, [2, 0, 1]], bg[:, None]], 1)
            value = pit_waveform_loss(estimates, refs, active, refs.sum(1) + bg, bg)
            self.assertLess(value['loss'].item(), 2e-6)

    def test_count_loss_requires_eligible_labels_and_has_no_gradient_for_others(self):
        logits = torch.randn(3, 3, requires_grad=True)
        counts = torch.tensor([1, 2, 3])
        loss = source_count_loss(logits, counts, torch.tensor([True, False, True]))
        loss.backward()
        self.assertFalse(logits.grad[1].any())
        self.assertTrue(logits.grad[[0, 2]].abs().sum() > 0)
        self.assertEqual(source_count_loss(logits, counts, torch.zeros(3, dtype=torch.bool)).item(), 0)
        with self.assertRaises(ValueError):
            source_count_loss(logits, counts, torch.ones(3))

    def test_full_backbone_three_encoders_sum_gradients_and_context_effect(self):
        # Small spatial geometry for a structural CPU check, never a reduced
        # channel model or a measurement of RF reconstruction performance.
        z = torch.randn(1, 32, 33, dtype=torch.complex64)
        context = torch.randn(1, 65, 256)
        start = torch.tensor([81920])
        for kind in ('tcn', 'transformer', 'lstm'):
            with self.subTest(kind=kind):
                model = ContextualSeparator(kind).eval()
                self.assertGreater(sum(p.numel() for p in model.parameters()), 31_000_000)
                result = model(z, context, start)
                self.assertEqual(result['count_logits'].shape, (1, 3))
                self.assertEqual(result['estimates'].shape, (1, 4, 32, 33))
                torch.testing.assert_close(result['estimates'].sum(1), z, atol=2e-6, rtol=2e-6)
                # Verify that reconstruction itself, not just the count head,
                # sends a nonzero finite gradient into the long-context branch.
                reconstruction_gradient = torch.autograd.grad(
                    result['estimates'].abs().square().mean(),
                    model.context_encoder.input.weight, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(reconstruction_gradient).all())
                self.assertGreater(reconstruction_gradient.abs().sum().item(), 0)
                (result['estimates'].abs().square().mean() + result['count_logits'].square().mean()).backward()
                for component in (model.context_encoder.input, model.context_projection, model.output):
                    self.assertTrue(torch.isfinite(component.weight.grad).all())
                    self.assertGreater(component.weight.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    changed = model(z, context.flip(-1), start)['estimates']
                self.assertGreater((changed - result['estimates']).abs().max().item(), 1e-7)
                del result, model

    def test_existing_backbone_weights_preserved_and_mean_control_order_invariant(self):
        plain = ComplexSeparator(3).eval()
        model = ContextualSeparator('tcn', context_mode='mean').eval()
        mismatch = model.load_state_dict(plain.state_dict(), strict=False)
        self.assertFalse(mismatch.unexpected_keys)
        self.assertTrue(all(k.startswith(('context_', 'count_head')) for k in mismatch.missing_keys))
        for name, tensor in plain.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[name], atol=0, rtol=0)
        z = torch.randn(1, 32, 33, dtype=torch.complex64)
        context = torch.randn(1, 65, 256)
        with torch.no_grad():
            a = model(z, context, torch.tensor([100000]))
            b = model(z, context.flip(-1), torch.tensor([100000]))
        torch.testing.assert_close(a['estimates'], b['estimates'], atol=2e-6, rtol=2e-6)
        with torch.no_grad():
            model.context_projection.weight.zero_()
            unconditioned = model(z, context, torch.tensor([100000]))['estimates']
            torch.testing.assert_close(unconditioned, plain(z), atol=0, rtol=0)
            # A crop ending exactly at the long-record boundary is legal;
            # center=True reflects its final STFT frame at that boundary.
            boundary = model(z, context, torch.tensor([2097152 - 4096]))['estimates']
            torch.testing.assert_close(boundary, plain(z), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'outside'):
            model(z, context, torch.tensor([2097100]))


if __name__ == '__main__':
    unittest.main()
