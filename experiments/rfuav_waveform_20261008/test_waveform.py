"""Checks for the long-I/Q contrast without reading held-out recordings."""
import unittest
import torch
from waveform_models import build, pack_iq, unpack_iq
from run_comparison import lr_for_update, TOTAL_UPDATES


class WaveformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_packing_is_exact_for_every_sample_and_phase(self):
        torch.manual_seed(2)
        z=torch.complex(torch.randn(2,8192),torch.randn(2,8192))
        self.assertTrue(torch.equal(unpack_iq(pack_iq(z)),z))
        # Every sample of a complex sinusoid is retained, not just one per block.
        t=torch.arange(8192)
        wave=torch.exp(2j*torch.pi*.31*t)[None]
        self.assertTrue(torch.equal(unpack_iq(pack_iq(wave)),wave))

    def test_invalid_packing_length_rejected(self):
        with self.assertRaises(ValueError): pack_iq(torch.ones(1,8191,dtype=torch.complex64))

    def test_full_capacity_and_exact_same_initial_weights(self):
        a=build('short_context'); b=build('long_context')
        count=sum(p.numel() for p in a.parameters())
        self.assertGreaterEqual(count,32142859)
        self.assertEqual(count,sum(p.numel() for p in b.parameters()))
        for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]),k)

    def test_lr_schedule_budget(self):
        self.assertAlmostEqual(lr_for_update(0),1e-4)
        self.assertAlmostEqual(lr_for_update(TOTAL_UPDATES-1),1e-5)
        values=[lr_for_update(i) for i in range(TOTAL_UPDATES)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        with self.assertRaises(ValueError): lr_for_update(TOTAL_UPDATES)


if __name__=='__main__': unittest.main()
