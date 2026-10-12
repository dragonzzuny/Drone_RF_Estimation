"""Mathematical checks, including a gradient equivalence and real slot switch."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/rfuav_dense_gated_20261008/vendor'))
from drone_rf.losses import pit_waveform_loss
from sequence_objective import assignments, fixed_loss, replay_bundle, assert_replay_safe


class Tiny(torch.nn.Module):
    # Tests algebra only; actual experiment uses the full 32M model.
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, dtype=torch.complex128))
        self.logits = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))

    def forward(self, x):
        return x[:,None]*self.w[None,:,None], self.logits[None]


def predict(model, item):
    return model(item['mixture'])


def data(count):
    refs = torch.randn(1,3,97,dtype=torch.complex128)
    refs[:,count:] = 0
    return dict(references=refs, mixture=refs.sum(1), active=torch.tensor([[i<count for i in range(3)]]),
                construction_count=torch.tensor([count]))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_independent_exactly_parent(self):
        for k in (1,2,3):
            item=data(k); y=torch.randn(1,4,97,dtype=torch.complex128,requires_grad=True)
            chosen,_=assignments([y],[item],'independent')
            original=pit_waveform_loss(y,item['references'],item['active'],item['mixture'])
            new=fixed_loss(y,item,chosen[0])
            torch.testing.assert_close(new,original['loss'],rtol=1e-13,atol=1e-13)
            g1=torch.autograd.grad(new,y,retain_graph=True)[0]
            g2=torch.autograd.grad(original['loss'],y)[0]
            torch.testing.assert_close(g1,g2,rtol=1e-13,atol=1e-13)

    def test_real_switch_is_penalized(self):
        item=data(2); y=torch.cat((item['references'],torch.zeros_like(item['mixture'][:,None])),1)
        local,diag=assignments([y,y[:,[1,0,2,3]]],[item,item],'independent')
        shared,other=assignments([y,y[:,[1,0,2,3]]],[item,item],'shared')
        self.assertNotEqual(local[0],local[1]); self.assertEqual(shared[0],shared[1])
        self.assertLess(abs(diag['local_min_loss']),1e-12)
        self.assertGreater(other['shared_min_loss'],.1)

    def test_replay_gradient_matches_joint_graph(self):
        for mode in ('independent','shared'):
            net=Tiny();assert_replay_safe(net);items=[data(k) for k in (1,2,3)]
            ys=[predict(net,item)[0] for item in items]
            orders,_=assignments(ys,items,mode)
            loss=sum(fixed_loss(y,it,p)+.1*torch.nn.functional.cross_entropy(net.logits[None],it['construction_count']-1)
                     for y,it,p in zip(ys,items,orders))/len(items)
            loss.backward(); expected=[p.grad.clone() for p in net.parameters()]
            net.zero_grad(set_to_none=True)
            replay_bundle(net,predict,items,mode,1)
            for p,g in zip(net.parameters(),expected):
                torch.testing.assert_close(p.grad,g,rtol=1e-12,atol=1e-12)

    def test_overlap_matching_is_reference_free_and_exact(self):
        from sequence_eval import merge, WINDOW
        rng=np.random.default_rng(8)
        reference=(rng.standard_normal((4,3*WINDOW))+1j*rng.standard_normal((4,3*WINDOW))).astype(np.complex64)
        starts=[j*(WINDOW//2) for j in range(5)]
        permutations=[(0,1,2),(2,0,1),(1,2,0),(2,1,0),(0,2,1)]
        predictions=[reference[[*p,3],s:s+WINDOW] for p,s in zip(permutations,starts)]
        result,_=merge(predictions,starts,3*WINDOW)
        # merge receives predictions only; references are used exclusively here.
        np.testing.assert_allclose(result,reference,rtol=1e-6,atol=1e-6)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
