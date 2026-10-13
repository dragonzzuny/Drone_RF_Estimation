"""Re-use all 100 U-Net B schedules exactly; never re-split or load old weights."""
import shutil
import os
import time
from data import *


def main():
    assert not (OUT/'READY.json').exists(), 'Preparation already completed'
    OUT.mkdir(parents=True,exist_ok=True);PREP.mkdir(exist_ok=True)
    write_json(OUT/'STATE.json',dict(status='PREPARING_MATCHED_SCHEDULES',pid=os.getpid(),time=time.time()))
    parent=ROOT/'local/bandwidth_timeblock_20261011_v1'
    extension=ROOT/'local/timeblock_extend100_20261012_v1'
    old=ROOT/'experiments/bandwidth_timeblock_20261011/data.py'
    assert (HERE/'data.py').read_text()==old.read_text().replace(
        'local/bandwidth_timeblock_20261011_v1','local/wavenet_timeblock100_20261013_v1')
    clips=read(MANIFEST)['clips'];validate_clips(clips)
    paths={};matched=[]
    for epoch in range(1,101):
        run=parent if epoch<=50 else extension
        event=read(run/f'EPOCH_{epoch:03d}.json');audit=read(run/f'AUDIT_{epoch:03d}.json')
        assert audit['status']=='PASS' and audit['event_sha256']==sha256(run/f'EPOCH_{epoch:03d}.json')
        name=f'{ROLES[0]}_{epoch:03d}.npy';source=run/'preparation'/name
        original=np.load(source,allow_pickle=False)
        assert np.array_equal(original,schedule(clips,ROLES[0],epoch,800))
        assert hashlib.sha256(original.tobytes()).hexdigest()==event['training_rows_sha256']
        shutil.copyfile(source,PREP/name);paths[name]=sha256(PREP/name)
        assert paths[name]==read(run/'preparation/PREPARATION.json')['schedules'][name]
        matched.append(dict(epoch=epoch,rows_sha256=event['training_rows_sha256']))
    name=f'{ROLES[1]}_001.npy';shutil.copyfile(parent/'preparation'/name,PREP/name)
    paths[name]=sha256(PREP/name)
    config=dict(read(parent/'preparation/PREPARATION.json'),schedules=paths,epochs=100,
                wavenet_comparison=True,parent_training_schedule_match=matched)
    write_json(PREP/'PREPARATION.json',config)
    stem=PREP/'features'/f'{ROLES[1]}_001';stem.parent.mkdir(exist_ok=True)
    source=parent/'preparation/features'/f'{ROLES[1]}_001.npy'
    if not stem.with_suffix('.npy').exists():os.link(source,stem.with_suffix('.npy'))
    receipt=read(source.with_suffix('.json'))
    assert sha256(stem.with_suffix('.npy'))==receipt['features_sha256']
    receipt.update(preparation_sha256=sha256(PREP/'PREPARATION.json'),reused_feature_bytes=True)
    write_json(stem.with_suffix('.json'),receipt)
    dev=Mixtures(ROLES[1])
    assert dev.rows_hash==read(parent/'PROTOCOL.json')['fixed_dev_schedule_sha256']
    write_json(OUT/'STATE.json',dict(status='PREPARING_FIRST_TRAIN_FEATURES',epoch=1,updates=0,
                                   pid=os.getpid(),time=time.time()))
    prepare_features(ROLES[0],1)
    train=Mixtures(ROLES[0],1)
    for k in (1,2,3):
        i=int(np.flatnonzero(train.rows['count']==k)[0]);fast,full=train[i],train.full_example(i)
        for key in ('mixture','references','context_features'):
            np.testing.assert_allclose(fast[key],full[key],rtol=2e-6,atol=2e-7)
    result=dict(status='PASS',matched_U_net_epochs=100,matched_examples_per_epoch=2400,
                total_optimizer_updates_planned=7500,loader_changed_only_output_path=True,
                development_feature_bytes_identical=True,
                fixed_dev_schedule_sha256=dev.rows_hash,long_short_synthesis_equivalence=True,
                heldout_access=False,old_waveform_weights_loaded=False,
                independent_recording_validation=False,preparation_sha256=sha256(PREP/'PREPARATION.json'))
    write_json(OUT/'READY.json',result)
    write_json(ROOT/'reports/2026-10-13/WAVENET100_PREPARATION.json',result)
    print('MATCHED_B100_PREPARATION_PASS',flush=True)


if __name__=='__main__':main()
