"""Check all frozen schedules, fresh capacity and nonidentity PIT scoring."""
import torch
from train import *


def main():
    torch.set_num_threads(1)
    for role, epochs in ((ROLES[0],50),(ROLES[1],1)):
        for epoch in range(1,epochs+1):
            dataset=Mixtures(role,epoch,features=False)
            for count in (1,2,3):
                assert int(np.sum(dataset.rows['count']==count))==(800 if role==ROLES[0] else 210)
    net=build('unet_mean')
    parameters=sum(p.numel() for p in net.parameters())
    assert parameters==32142859 and all(p.requires_grad for p in net.parameters())
    assert not list(net.buffers())
    rng=torch.Generator().manual_seed(0)
    refs=torch.complex(torch.randn(1,3,512,generator=rng),torch.randn(1,3,512,generator=rng))
    mixture=refs.sum(1)
    estimates=torch.cat((refs[:,[2,0,1]],torch.zeros_like(refs[:,:1])),1)
    scores=waveform_metrics(estimates,refs,torch.ones(1,3,dtype=torch.bool),mixture)
    assert float(scores['nmse'].max())==0 and scores['assignment'].tolist()==[[1,2,0]]
    report=dict(status='PASS',all_50_training_schedules_checked=True,fixed_dev_schedule_checked=True,
        full_parameters=parameters,all_parameters_trainable=True,nonidentity_pit_recovery_nmse=0.,
        old_checkpoint_loaded=False,preparation_sha256=sha256(PREP/'PREPARATION.json'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (HERE/'check_cpu.py',HERE/'train.py',HERE/'data.py')})
    write_json(OUT/'CPU_CHECK.json',report)
    write_json(PUBLIC/'TIMEBLOCK_CPU_CHECK.json',report)
    print('TIMEBLOCK_CPU_CHECK_PASS',flush=True)


if __name__=='__main__':main()
