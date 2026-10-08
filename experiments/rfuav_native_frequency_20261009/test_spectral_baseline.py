"""Mixture-only filter regression on separate synthetic frequency components."""
import unittest
import numpy as np
import torch
from spectral_baseline import profile,infer


class SpectralBaselineTests(unittest.TestCase):
    def test_recovers_nonoverlapping_tones_and_preserves_sum(self):
        torch.set_num_threads(2)
        n=np.arange(131072)
        a=np.exp(2j*np.pi*43*n/512).astype(np.complex64)
        b=np.exp(-2j*np.pi*77*n/512).astype(np.complex64)
        powers=[profile(v) for v in (a,b)]
        bank=dict(psd=[(p/p.sum()).tolist() for p in powers])
        refs=np.asarray([np.sqrt(.7)*a,np.sqrt(.3)*b],dtype=np.complex64)
        mix=refs.sum(0)
        output,count,fractions,_=infer(mix[:63872],profile(mix),bank)
        self.assertEqual(count,2)
        np.testing.assert_allclose(fractions,[.7,.3],atol=1e-5)
        value=output[0].numpy()
        for i in range(2):
            self.assertLess(np.mean(abs(value[i]-refs[i,:63872])**2)/np.mean(abs(refs[i,:63872])**2),.002)
        self.assertLess(np.mean(abs(value.sum(0)-mix[:63872])**2),1e-12)
        self.assertEqual(np.max(abs(value[2:])),0.)


if __name__=='__main__':unittest.main()
