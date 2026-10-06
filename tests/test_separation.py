import unittest
import torch
from drone_rf.losses import pit_waveform_loss
from drone_rf.model import ComplexSeparator


class SeparationChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_permuted_perfect_sources_and_silent_slots(self):
        ref=torch.randn(2,4,257,dtype=torch.complex64)
        ref[0,2:]=0
        active=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        bg=.03*torch.randn(2,257,dtype=torch.complex64)
        mix=ref.sum(1)+bg
        est=torch.cat([ref[:,[3,1,0,2]],bg[:,None]],1)
        value=pit_waveform_loss(est,ref,active,mix,bg)
        self.assertLess(value['loss'].item(),2e-6)

    def test_extra_source_is_penalized_and_gradient_finite(self):
        ref=torch.randn(1,2,257,dtype=torch.complex64);ref[:,1]=0
        active=torch.tensor([[True,False]])
        est=torch.cat([ref,torch.zeros_like(ref[:,:1])],1)
        est[:,1]=.2*ref[:,0];est[:,0]*=.8
        est.requires_grad_()
        value=pit_waveform_loss(est,ref,active,ref.sum(1))
        self.assertGreater(value['loss'].item(),.01)
        value['loss'].backward()
        self.assertTrue(torch.isfinite(est.grad).all())

    def test_target_permutation_does_not_change_loss(self):
        ref=torch.randn(3,4,129,dtype=torch.complex64)
        est=torch.randn(3,5,129,dtype=torch.complex64,requires_grad=True)
        active=torch.ones(3,4,dtype=torch.bool);mix=ref.sum(1)
        a=pit_waveform_loss(est,ref,active,mix)['loss']
        b=pit_waveform_loss(est,ref[:,[2,0,3,1]],active,mix)['loss']
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_fixed_order_control_penalizes_a_swapped_pair(self):
        ref=torch.randn(1,2,129,dtype=torch.complex64)
        est=torch.cat([ref.flip(1),torch.zeros_like(ref[:,:1])],1)
        active=torch.ones(1,2,dtype=torch.bool);mix=ref.sum(1)
        pit=pit_waveform_loss(est,ref,active,mix)['loss']
        fixed=pit_waveform_loss(est,ref,active,mix,assignment_mode='fixed')['loss']
        self.assertLess(pit.item(),2e-6)
        self.assertGreater(fixed.item(),.1)

    def test_no_drone_with_background(self):
        ref=torch.zeros(1,4,257,dtype=torch.complex64)
        bg=torch.randn(1,257,dtype=torch.complex64)
        est=torch.cat([ref,bg[:,None]],1)
        value=pit_waveform_loss(est,ref,torch.zeros(1,4,dtype=torch.bool),bg,bg)
        self.assertEqual(value['loss'].item(),0)

    def test_full_network_sum_and_nonreduced_body(self):
        net=ComplexSeparator(4)
        self.assertGreater(sum(p.numel() for p in net.parameters()),31_000_000)
        z=torch.randn(1,32,33,dtype=torch.complex64)
        result=net(z)
        self.assertEqual(result.shape,(1,5,32,33))
        self.assertTrue(torch.allclose(result.sum(1),z,atol=2e-6,rtol=2e-6))
        result.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(net.output.weight.grad).all())


if __name__=='__main__':unittest.main()
