"""Check exact optimizer/RNG handoff without reading or fitting RF waveforms."""
import ast
from pathlib import Path
import tempfile
import torch
import continue_low_lr as worker
import watch_epochs as watch


def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    p=torch.nn.Parameter(torch.tensor([1.,-.5],dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=1e-4,foreach=False)
    for _ in range(75):
        p.grad=torch.tensor([.25,-.1],dtype=p.dtype);opt.step()
    state=dict(model={'parameter':p.detach().clone()},optimizer=opt.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=[],epoch=1,updates=75,
        best={'epoch':1,'metric':.6},arm='original',protocol_sha256='old')
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'source.pt';target=Path(directory)/'target.pt'
        torch.save(state,source)
        worker.derive_checkpoint(source,target,'old','new')
        derived=torch.load(target,map_location='cpu',weights_only=False)
        assert derived['protocol_sha256']=='new' and derived['imported_from']['new_training_updates']==0
        q=torch.nn.Parameter(derived['model']['parameter'].clone())
        continuation=torch.optim.AdamW([q],lr=1e-4,weight_decay=1e-4,foreach=False)
        continuation.load_state_dict(derived['optimizer'])
        assert worker.identical(derived['torch_rng'],state['torch_rng'])
        for _ in range(3):
            for parameter,optimizer in ((p,opt),(q,continuation)):
                parameter.grad=torch.tensor([-.2,.3],dtype=parameter.dtype);optimizer.step()
        assert torch.equal(p,q)
        assert worker.identical(opt.state_dict(),continuation.state_dict())
        rejected=False
        try:worker.derive_checkpoint(source,target,'wrong','new')
        except ValueError:rejected=True
        assert rejected
    path=Path(worker.__file__).resolve()
    tree=ast.parse(path.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    text=ast.unparse(run)
    assert 'for epoch in (2, 3)' in text
    assert 'prior.train_epoch(root, arm, epoch, plan, train, validation, identities)' in text
    assert 'list(reversed' in text
    old=watch.read(watch.ROOT/'reports/2026-10-09/LOW_LR_UNET_CPU_CHECK.json')
    assert old['status']=='PASS'
    assert watch.digest(Path(worker.prior.__file__))==old['worker_sha256']
    report=dict(status='PASS',worker_sha256=watch.digest(path),source_sha256={
        str(Path(__file__).relative_to(watch.ROOT)):watch.digest(Path(__file__))},
        unchanged_training_function_sha256=watch.digest(Path(worker.prior.__file__)),
        exact_state_roundtrip=True,continued_adam_matches_uninterrupted=True,
        invalid_protocol_rejected=True,imported_epochs=[1],executed_epochs=[2,3],
        imported_updates_per_arm=75,executed_updates_per_arm=150,
        rf_training_updates=0,recorded_iq_reads=0,
        note='Synthetic optimizer state test, not a reduced separator or RF performance experiment.')
    watch.write(worker.CHECK,report);print(report)


if __name__=='__main__':main()
