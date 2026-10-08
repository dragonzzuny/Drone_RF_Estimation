import unittest
import torch
from diagnose_reference_tf_power import reference_masks


class ReferencePowerTests(unittest.TestCase):
    def test_partition_and_inactive_slot_including_empty_bins(self):
        z=torch.zeros(1,3,4,8,dtype=torch.complex64)
        z[:,0,0]=2;z[:,1,1]=3
        for weights in reference_masks(z,2).values():
            self.assertTrue(torch.allclose(weights.sum(1),torch.ones(1,4,8)))
            self.assertEqual(float(weights[:,2].abs().max()),0.)
            self.assertTrue(torch.equal(weights[:,0,0],torch.ones(1,8)))
            self.assertTrue(torch.equal(weights[:,1,1],torch.ones(1,8)))

    def test_phase_invariant_power_and_time_local_allocation(self):
        z=torch.zeros(1,3,2,8,dtype=torch.complex64)
        z[:,0,:,:4]=1;z[:,1,:,4:]=1
        a=reference_masks(z,2);b=reference_masks(1j*z,2)
        for key in a:
            self.assertTrue(torch.equal(a[key],b[key]))
        self.assertTrue(torch.equal(a['reference_frequency_mean_power'][:,:2],torch.full((1,2,2,8),.5)))
        self.assertTrue(torch.equal(a['reference_time_frequency_power'][:,0,:,:4],torch.ones(1,2,4)))


if __name__=='__main__':
    unittest.main()
