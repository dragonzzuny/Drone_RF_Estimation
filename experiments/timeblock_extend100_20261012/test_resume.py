"""Optimizer/scheduler/RNG continuation must match uninterrupted updates exactly."""
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import handoff
from handoff import restore, exact


class ResumeTest(unittest.TestCase):
    def test_next_update_matches_with_reduced_lr_and_dropout(self):
        torch.set_num_threads(1)
        torch.manual_seed(79)
        net = torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Dropout(.3),torch.nn.Linear(7,2))
        opt = torch.optim.AdamW(net.parameters(),lr=.01,foreach=False)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=0)
        x = torch.arange(24,dtype=torch.float32).reshape(6,4)/24
        def step(n,o):
            o.zero_grad(set_to_none=True)
            n(x).square().mean().backward()
            o.step()
        for _ in range(3):
            step(net,opt)
        sched.step(1.)
        sched.step(2.)
        assert opt.param_groups[0]['lr'] == .005
        buf = io.BytesIO()
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),torch_rng=torch.get_rng_state()),buf)
        buf.seek(0)
        saved = torch.load(buf,weights_only=False)
        step(net,opt)
        sched.step(3.)
        restarted = torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Dropout(.3),torch.nn.Linear(7,2))
        other = torch.optim.AdamW(restarted.parameters(),lr=.01,foreach=False)
        other_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(other,factor=.5,patience=0)
        restore(saved,restarted,other,other_sched,cuda=False)
        step(restarted,other)
        other_sched.step(3.)
        exact(net.state_dict(),restarted.state_dict())
        exact(opt.state_dict(),other.state_dict())
        exact(sched.state_dict(),other_sched.state_dict())

    def test_wait_requires_completed_epoch50(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with patch.object(handoff,'PARENT',parent), \
                 patch.object(handoff,'verify_parent'), \
                 patch.object(handoff,'audited_event') as audited, \
                 patch.object(handoff,'read',return_value=dict(
                     status='COMPLETE_AUDITED',epoch=50,updates=3750,pid=99999999)):
                handoff.wait_parent(lambda *a,**k:None,lambda:False)
                audited.assert_called_once_with(parent,50)
            with patch.object(handoff,'PARENT',parent), \
                 patch.object(handoff,'verify_parent'), \
                 patch.object(handoff,'read',return_value=dict(
                     status='COMPLETE_AUDITED',epoch=49,updates=3675,pid=99999999)):
                with self.assertRaises(AssertionError):
                    handoff.wait_parent(lambda *a,**k:None,lambda:False)

    def test_dead_parent_cannot_trigger_extension(self):
        with patch.object(handoff,'verify_parent'), \
             patch.object(handoff,'read',return_value=dict(
                 status='FAILED',epoch=49,updates=3675,pid=99999999)):
            with self.assertRaises(AssertionError):
                handoff.wait_parent(lambda *a,**k:None,lambda:False)


if __name__ == '__main__':
    unittest.main()
