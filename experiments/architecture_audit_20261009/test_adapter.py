"""Full model capacity, bounded synthetic inputs for implementation checks."""
import unittest
import torch
from native_wavenet import build_native_wavenet
from models import build, predict
from drone_rf.losses import pit_waveform_loss

torch.set_num_threads(2)


class NativeAdapterTests(unittest.TestCase):
    def test_historical_forward_preserved(self):
        old = build('wavenet').eval()
        new = build_native_wavenet().eval()
        self.assertEqual(sum(p.numel() for p in new.parameters()), 4601867)
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        torch.manual_seed(123)
        iq = torch.randn(1, 769, dtype=torch.complex64)
        features = torch.randn(1, 65, 256)
        start = torch.tensor([10000])
        with torch.no_grad():
            a, b = old(iq, features, start), new(iq, features, start)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_native_checkpoint_and_input_whitelist(self):
        net = build_native_wavenet().train()
        torch.manual_seed(124)
        ref = torch.randn(1, 3, 769, dtype=torch.complex64)
        item = dict(mixture=ref.sum(1), references=ref, active=torch.ones(1, 3, dtype=torch.bool),
                    context_features=torch.randn(1, 65, 255), crop_start=torch.tensor([2000]))
        outputs, gradients = [], []
        for enabled in (False, True):
            net.checkpoint_blocks = enabled
            net.zero_grad(set_to_none=True)
            output, _ = predict(net, item)
            loss = pit_waveform_loss(output, ref, item['active'], item['mixture'])['loss']
            loss.backward()
            outputs.append(output.detach())
            gradients.append(net.blocks[0].filter_gate.weight.grad.clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-6, atol=1e-8)
        self.assertTrue(torch.isfinite(gradients[1]).all())
        self.assertGreater(float(gradients[1].abs().sum()), 0)
        with torch.no_grad():
            unlabeled = {k: item[k] for k in ('mixture', 'context_features', 'crop_start')}
            output, _ = predict(net.eval(), unlabeled)
        torch.testing.assert_close(output, outputs[-1], rtol=0, atol=0)
        torch.testing.assert_close(output.sum(1), item['mixture'], rtol=2e-5, atol=2e-6)
        with self.assertRaises(ValueError):
            net(item['mixture'], item['context_features'], torch.tensor([255*8192-100]))


if __name__ == '__main__':
    unittest.main()
