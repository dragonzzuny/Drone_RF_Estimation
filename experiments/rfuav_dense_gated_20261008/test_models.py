"""Implementation checks on full parameter capacity; short inputs bound CPU cost."""
import gc
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import torch

from models import ARMS, GatedUNet, WaveNetSeparator, build, predict
from study import run, DEFAULT_PREPARATION
from drone_rf.losses import pit_waveform_loss
from drone_rf.mixture_constraints import validate_sources


torch.set_num_threads(2)


def example(count=3, length=2048):
    generator = torch.Generator().manual_seed(901)
    refs = torch.complex(torch.randn(1, 3, length, generator=generator),
                         torch.randn(1, 3, length, generator=generator))
    refs[:, count:] = 0
    return dict(mixture=refs.sum(1), references=refs, active=torch.arange(3)[None] < count,
                context_features=torch.randn(1, 65, 256, generator=generator),
                crop_start=torch.tensor([10000]), construction_count=torch.tensor([count]))


class ArchitectureTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_same_initial_unet_and_gate_learning(self):
        base, gated = build('unet_mean'), build('unet_gated')
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[name]), name)
        item = example()
        with torch.no_grad():
            before, logits = predict(base.eval(), item)
            actual, actual_logits = predict(gated.eval(), item)
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, logits, rtol=0, atol=0)
        del base
        optimizer = torch.optim.SGD(gated.parameters(), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output, _ = predict(gated.train(), item)
            loss = pit_waveform_loss(output, item['references'], item['active'], item['mixture'])['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(gated.gates.expand.weight.grad.abs().sum()), 0)
            if step:
                self.assertGreater(float(gated.gates.reduce.weight.grad.abs().sum()), 0)
            optimizer.step()
        with torch.no_grad():
            after, _ = predict(gated.eval(), item)
        self.assertGreater(float((after - before).abs().max()), 0)

    def test_checkpoint_preserves_wavenet_outputs_and_gradients(self):
        net = build('wavenet').train()
        item = example(length=769)
        outputs = []
        gradients = []
        for enabled in (False, True):
            net.checkpoint_blocks = enabled
            net.zero_grad(set_to_none=True)
            output, _ = predict(net, item)
            loss = pit_waveform_loss(output, item['references'], item['active'], item['mixture'])['loss']
            loss.backward()
            outputs.append(output.detach())
            gradients.append(net.blocks[0].filter_gate.weight.grad.clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-6, atol=1e-8)
        self.assertGreater(float(gradients[1].abs().sum()), 0)
        self.assertEqual(len(net.blocks), 30)
        self.assertEqual(net.input_projection.out_channels, 128)

    def test_all_counts_mixture_projection_and_input_only_prediction(self):
        for arm in ARMS:
            net = build(arm).eval()
            for count in (1, 2, 3):
                item = example(count=count)
                with torch.no_grad():
                    output, logits = predict(net, item)
                    unlabelled = {k: item[k] for k in ('mixture', 'context_features', 'crop_start')}
                    other, other_logits = predict(net, unlabelled)
                    loss = pit_waveform_loss(output, item['references'], item['active'], item['mixture'])['loss']
                self.assertEqual(tuple(output.shape), (1, 4, 2048))
                self.assertEqual(tuple(logits.shape), (1, 3))
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(output.sum(1), item['mixture'], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(output, other, rtol=0, atol=0)
                torch.testing.assert_close(logits, other_logits, rtol=0, atol=0)
            del net
            gc.collect()

    def test_pit_handles_permutations_and_missing_sources(self):
        for count in (1, 2, 3):
            item = example(count)
            refs = item['references']
            out = torch.cat((refs[:, [2, 0, 1]], torch.zeros_like(refs[:, :1])), 1)
            first = pit_waveform_loss(out, refs, item['active'], item['mixture'])
            second = pit_waveform_loss(out, refs[:, [1, 2, 0]], item['active'][:, [1, 2, 0]], item['mixture'])
            self.assertLess(float(first['loss']), 1e-5)
            torch.testing.assert_close(first['loss'], second['loss'])

    def test_cross_band_admission_rejected(self):
        a = dict(dataset='RFUAV', center_hz=2.45e9, fs_hz=1e8)
        validate_sources([a, dict(a, center_hz=2.47e9)])
        with self.assertRaises(ValueError):
            validate_sources([a, dict(a, center_hz=5.8e9)])
        with self.assertRaises(ValueError):
            validate_sources([a, dict(a, dataset='different')])

    def test_gpu_gate_stops_before_reading_source_iq(self):
        with TemporaryDirectory() as root:
            args = SimpleNamespace(run=Path(root), preparation=DEFAULT_PREPARATION,
                                   preflight_only=True, epochs=5)
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.cuda.device_count', return_value=0), \
                 patch('study.admitted_dataset', side_effect=AssertionError('Unexpected source access')):
                self.assertEqual(run(args), 2)
            receipt = json.loads((Path(root) / 'BLOCKED_GPU.json').read_text())
            self.assertEqual(receipt['status'], 'NOT_STARTED_CUDA_UNAVAILABLE')
            self.assertFalse(receipt['source_data_read'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
