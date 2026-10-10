"""Numerical, permutation, no-label-input and full-size GPU checks."""
import argparse
import copy
import fcntl
import time
import torch
from torch import nn
from common import ROOT, PUBLIC, PARENT, PARENT_SHA, PREPARATION, worker, w, eight
import objective as obj

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(8, 2, dtype=torch.float64) * .2)
        self.bias = nn.Parameter(torch.randn(8, dtype=torch.float64) * .1)
        self.logits = nn.Parameter(torch.randn(1, 3, dtype=torch.float64))
    def forward(self, item):
        assert set(item) <= {'mixture', 'context_features', 'crop_start'}
        x = item['mixture']; iq = torch.stack((x.real, x.imag), 1)
        y = torch.tanh(torch.einsum('oi,bit->bot', self.matrix, iq) + self.bias[None, :, None])
        z = torch.complex(y[:, 0::2], y[:, 1::2])
        z = z + (x - z.sum(1))[:, None] / 4
        return z, self.logits

def toy_predict(net, item):
    return net(item)

def cpu():
    torch.set_num_threads(2); torch.manual_seed(73)
    refs = torch.randn(1, 3, 257, dtype=torch.complex128)
    item = dict(references=refs, mixture=refs.sum(1), active=torch.ones(1, 3, dtype=torch.bool),
                construction_count=torch.tensor([3]), context_features=torch.zeros(1, 65, 255),
                crop_start=torch.tensor([0]))
    rows = []
    for k in (1, 2, 3):
        current = dict(item, references=refs.clone(), active=torch.arange(3)[None] < k,
                       construction_count=torch.tensor([k]))
        current['references'][:, k:] = 0
        current['mixture'] = current['references'].sum(1)
        net = Toy(); replay = copy.deepcopy(net)
        views, logits, orders = obj.collect(net, toy_predict, current)
        loss, detail = obj.losses(views, logits, current)
        loss.backward()
        repeated = obj.backward_replay(replay, toy_predict, current, check_replay=True)
        maximum = 0.
        for a, b in zip(net.parameters(), replay.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-11)
            maximum = max(maximum, float((a.grad-b.grad).abs().max()))
        with torch.no_grad():
            _, expected, _, _ = eight.predict_eight(net, toy_predict, current)
            torch.testing.assert_close(obj.average(views), expected, rtol=0, atol=0)
        correct = torch.cat((current['references'], torch.zeros_like(current['mixture'][:, None])), 1)
        ideal = correct[None].expand(8, -1, -1, -1)
        _, ideal_detail = obj.losses(ideal, logits, current)
        assert ideal_detail['consistency'] == 0 and abs(ideal_detail['waveform']) < 1e-10
        wrong = ideal * 0
        _, wrong_detail = obj.losses(wrong, logits, current)
        assert wrong_detail['consistency'] == 0 and wrong_detail['waveform'] > .5
        rows.append(dict(count=k,maximum_gradient_error=maximum,loss=detail,
                         replay_loss=repeated['total'],orders=orders,wrong_equal_output_loss=wrong_detail['waveform']))
    out=dict(status='PASS',tests=rows,full_vs_replay_gradients=True,
             selected_eight_inference_identical=True,wrong_identical_outputs_penalized=True,
             source_sha256={str(p.relative_to(ROOT)):w.digest(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')})
    w.write(PUBLIC/'PHASE_CONSISTENT_CPU_CHECK.json',out)
    print(out,flush=True)

def gpu():
    torch.set_num_threads(2); torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    assert w.digest(PARENT)==PARENT_SHA and torch.cuda.is_available()
    data=worker.NativeMixtures(PREPARATION,'train_pack',1)
    indices=[next(i for i,r in enumerate(data.rows) if int(r['count'])==k) for k in (1,2,3)]
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        net=worker.make_model('retained_unet',PARENT).cuda().train()
        assert sum(p.numel() for p in net.parameters())==32142859
        assert not any(isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)) for m in net.modules())
        assert not list(net.buffers())
        opt=torch.optim.AdamW(net.parameters(),lr=1e-6,weight_decay=1e-4,foreach=False)
        torch.cuda.reset_peak_memory_stats(); began=time.time();rows=[]
        opt.zero_grad(set_to_none=True)
        for i in indices:
            item=worker.fit.base.batch([data[i]])
            t=time.time()
            r=obj.backward_replay(net,worker.predict,item,divisor=3,check_replay=True)
            torch.cuda.synchronize()
            rows.append(dict(index=i,count=int(item['construction_count'][0]),seconds=time.time()-t,loss=r))
        norms={}
        for name in ('down','up','output','context_encoder','context_projection','count_head'):
            parts=[p.grad.square().sum() for n,p in net.named_parameters() if n.startswith(name+'.') and p.grad is not None]
            value=float(torch.stack(parts).sum().sqrt());assert value>0 and __import__('math').isfinite(value)
            norms[name]=value
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
        assert {int(v['step']) for v in opt.state.values()}=={1}
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        receipt=dict(status='PASS',parameters=32142859,trainable_parameters=32142859,
            rows=rows,gradient_norms=norms,preclip_norm=norm,seconds=time.time()-began,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),discarded_updates=1,
            parent_sha256=PARENT_SHA,source_sha256={str(p.relative_to(ROOT)):w.digest(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')})
        w.write(PUBLIC/'PHASE_CONSISTENT_GPU_CHECK.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
    gpu() if a.gpu else cpu()
