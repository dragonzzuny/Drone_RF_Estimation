"""Check full pretrained state mapping and augmented checkpoint reconstruction."""
import gc
from pathlib import Path
import torch
import train_comparison as worker


def main():
    torch.set_num_threads(2)
    w=worker.watch
    receipt=w.read(worker.ROOT/'local/native_frequency_20261009_v1/phase/CHECKPOINT.json')
    parent=Path(receipt['path'])
    assert w.digest(parent)==receipt['sha256'] and receipt['selected_epoch']==2
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    results=[]
    for arm in worker.ARMS:
        net=worker.make_model(arm,parent).eval()
        state=net.state_dict()
        for key,value in saved['model'].items():
            mapped=key.replace('output.','output.base.',1) if arm=='source_interaction' and key.startswith('output.') else key
            assert torch.equal(value,state[mapped]),key
        parameters=sum(p.numel() for p in net.parameters())
        expected=32_180_747 if arm=='source_interaction' else 32_142_859
        assert parameters==expected
        if arm=='source_interaction':
            assert torch.count_nonzero(net.output.readout.weight)==0
            # A nonzero trained head must also roundtrip; all-zero would hide
            # lost new parameters when constructing a resumed architecture.
            torch.nn.init.normal_(net.output.readout.weight,std=.02)
            rebuilt=worker.make_model(arm).eval()
            rebuilt.load_state_dict(net.state_dict(),strict=True)
            features=torch.randn(1,64,17,19)
            with torch.no_grad():
                torch.testing.assert_close(net.output(features),rebuilt.output(features),rtol=0,atol=0)
            del rebuilt,features
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)
        assert len(optimizer.param_groups)==1 and optimizer.param_groups[0]['lr']==1e-5
        assert sum(p.numel() for p in optimizer.param_groups[0]['params'])==parameters
        assert all(p.requires_grad for p in net.parameters())
        results.append(dict(arm=arm,parameters=parameters,parent_state_exact=True,
                            all_parameters_trainable=True,learning_rate=1e-5))
        del state,net,optimizer
        gc.collect()
    result=dict(status='PASS',worker_sha256=w.digest(Path(worker.__file__)),
        checker_sha256=w.digest(Path(__file__)),parent_checkpoint_sha256=receipt['sha256'],
        source_head_sha256=w.digest(Path(__file__).with_name('source_head.py')),
        results=results,nonzero_augmented_head_checkpoint_roundtrip=True,
        recorded_iq_reads=0,heldout_read=False,training_updates=0)
    w.write(worker.CONTROL_CHECK,result);print(result)


if __name__=='__main__':main()
