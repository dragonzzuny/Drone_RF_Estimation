"""Geometry and initialization checks without loading any recording I/Q."""
import hashlib
import unittest
import torch
from fusion_models import build, align_iq_context, CONTEXT_STEP, CONTEXT_FIRST, PARENT, PARENT_SHA256


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_alignment_matches_native_sample_coordinates(self):
        values = (CONTEXT_FIRST + CONTEXT_STEP * torch.arange(256, dtype=torch.float32))[None, None]
        for frames, first, stride in ((65536, 7.5, 16), (256, 2047.5, 4096)):
            aligned = align_iq_context(values, frames, first, stride)
            expected = (first + stride * torch.arange(frames, dtype=torch.float32)).clamp(values.min(), values.max())
            self.assertLess(float((aligned[0, 0] - expected).abs().max()), .2)

    def test_same_full_model_and_exact_common_parent(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA256)
        a, b = build('local_only'), build('local_global')
        self.assertGreater(sum(p.numel() for p in a.parameters()), 32355203)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[name]), name)
        parent = torch.load(PARENT, map_location='cpu', weights_only=False)['model']
        for name, value in parent.items():
            self.assertTrue(torch.equal(value, a.state_dict()[name]), name)

    def test_optimizer_covers_all_parameters_and_fixed_lr_budget(self):
        from run_fusion import optimizer_for, lr_for_update, TOTAL_UPDATES
        net = build('local_only')
        optimizer = optimizer_for(net)
        identifiers = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(p) for p in net.parameters()})
        self.assertEqual([g['lr_multiplier'] for g in optimizer.param_groups], [1., 10.])
        self.assertAlmostEqual(lr_for_update(0), 1e-5)
        self.assertAlmostEqual(lr_for_update(TOTAL_UPDATES - 1), 1e-6)


if __name__ == '__main__':
    unittest.main()
