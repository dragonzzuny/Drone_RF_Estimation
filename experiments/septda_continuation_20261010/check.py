"""No-update CPU check of both real model/optimizer/RNG resume states."""
import gc
from pathlib import Path
import torch
import train as c

torch.set_num_threads(2)
rows = []
for arm, folder in c.ORIGINS.items():
    net, optimizer, best, monitor, proof = c.restore(arm, folder / 'LAST.pt', 'cpu', 1)
    assert best['epoch'] == 0 and monitor == dict(best=None, bad=0, reductions=0)
    assert len(optimizer.param_groups) == (2 if arm == 'septda' else 1)
    assert [g['lr'] for g in optimizer.param_groups] == ([1e-5, 1e-4] if arm == 'septda' else [1e-5])
    assert all(g['weight_decay'] == 1e-4 for g in optimizer.param_groups)
    rows.append(dict(arm=arm, parameters=sum(p.numel() for p in net.parameters()),
                     optimizer_states=len(optimizer.state), **proof))
    del net, optimizer; gc.collect()
assert [(e + 1) % 5 + 1 for e in range(1, 7)] == [3, 4, 5, 1, 2, 3]
assert not torch.cuda.is_initialized()
report = dict(status='PASS', rows=rows, worker_sha256=c.w.digest(Path(c.__file__)),
              checker_sha256=c.w.digest(Path(__file__)), new_updates=0, recorded_iq_reads=0,
              cuda_initialized=False, heldout_read=False)
c.w.write(c.PUBLIC / 'SEPTDA_RF_CONTINUATION_CHECK.json', report)
print(report, flush=True)
