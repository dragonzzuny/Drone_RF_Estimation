import unittest
import torch
from allocation import allocation_kl


class AllocationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.refs=torch.randn(1,3,2048,dtype=torch.complex64)
        self.order=torch.tensor([[0,1,2]])
        self.exact=torch.cat((self.refs,torch.zeros_like(self.refs[:,:1])),1)

    def test_exact_allocation_and_permutation(self):
        loss=allocation_kl(self.exact,self.refs,self.order)
        permuted=self.exact[:,[2,0,1,3]]
        changed=allocation_kl(permuted,self.refs,torch.tensor([[1,2,0]]))
        self.assertLess(abs(float(loss)),1e-6)
        self.assertTrue(torch.allclose(loss,changed,atol=1e-7))

    def test_common_phase_scale_and_batch_do_not_change_loss(self):
        wrong=self.exact*torch.tensor([.3,1.4,.6,1.])[None,:,None]
        baseline=allocation_kl(wrong,self.refs,self.order)
        changed=allocation_kl(wrong*(3j),self.refs*(3j),self.order)
        stacked=allocation_kl(wrong.repeat(2,1,1),self.refs.repeat(2,1,1),self.order.repeat(2,1))
        self.assertTrue(torch.allclose(baseline,changed,rtol=1e-5,atol=1e-7))
        self.assertTrue(torch.allclose(baseline,stacked,rtol=1e-6,atol=1e-7))

    def test_wrong_allocation_and_empty_slot_have_finite_gradient(self):
        wrong=self.exact.clone();wrong[:,0]*=.3;wrong[:,2]=0;wrong.requires_grad_(True)
        loss=allocation_kl(wrong,self.refs,self.order);loss.backward()
        self.assertGreater(float(loss.detach()),.01)
        self.assertTrue(torch.isfinite(wrong.grad).all())
        self.assertGreater(float(wrong.grad.abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
