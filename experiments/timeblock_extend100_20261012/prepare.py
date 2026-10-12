"""Continue identical deterministic schedules to e100; reuse unchanged fixed DEV."""
import importlib.util
import gc
import shutil
import torch
from data import *
from handoff import PARENT, PUBLIC, verify_parent, restore


def main():
    parent = verify_parent()
    PREP.mkdir(exist_ok=True)
    clips = read(MANIFEST)['clips']
    validate_clips(clips)
    # The only data-loader change is the run directory, checked byte for byte.
    old = ROOT/'experiments/bandwidth_timeblock_20261011/data.py'
    expected = old.read_text().replace('local/bandwidth_timeblock_20261011_v1',
                                      'local/timeblock_extend100_20261012_v1')
    assert (HERE/'data.py').read_text() == expected
    previous = read(PARENT/'preparation/PREPARATION.json')
    paths = {}
    for epoch in range(51,101):
        rows = schedule(clips,ROLES[0],epoch,800)
        assert np.array_equal(rows,schedule(clips,ROLES[0],epoch,800))
        assert all(int(np.sum(rows['count']==k)) == 800 for k in (1,2,3))
        path = PREP/f'{ROLES[0]}_{epoch:03d}.npy'
        if path.exists():
            assert np.array_equal(np.load(path,allow_pickle=False),rows)
        else:
            np.save(path,rows,allow_pickle=False)
        paths[path.name] = sha256(path)
    devname = f'{ROLES[1]}_001.npy'
    shutil.copyfile(PARENT/'preparation'/devname, PREP/devname)
    assert sha256(PREP/devname) == previous['schedules'][devname]
    paths[devname] = sha256(PREP/devname)
    config = dict(previous, schedules=paths, epochs=100, first_training_epoch=51,
        continuation_of_same_B_model=True, parent_preparation_sha256=sha256(PARENT/'preparation/PREPARATION.json'))
    write_json(PREP/'PREPARATION.json',config)
    # Same schedule, same feature bytes. Receipt belongs to the new preparation.
    devstem = PREP/'features'/f'{ROLES[1]}_001'
    devstem.parent.mkdir(exist_ok=True)
    src = PARENT/'preparation/features'/f'{ROLES[1]}_001.npy'
    if not devstem.with_suffix('.npy').exists():
        os.link(src,devstem.with_suffix('.npy'))
    receipt = read(src.with_suffix('.json'))
    assert sha256(devstem.with_suffix('.npy')) == receipt['features_sha256']
    receipt.update(preparation_sha256=sha256(PREP/'PREPARATION.json'), reused_parent_feature_bytes=True)
    write_json(devstem.with_suffix('.json'),receipt)
    dev = Mixtures(ROLES[1])
    assert dev.rows_hash == parent['fixed_dev_schedule_sha256']
    for epoch in range(51,101):
        Mixtures(ROLES[0],epoch,features=False)
    del dev
    # Full-size CPU checkpoint round trip on a stable currently audited LAST.
    # This is a preflight only; actual continuation waits for audited epoch50.
    from models import build
    torch.set_num_threads(1)
    saved = torch.load(PARENT/'LAST.pt',map_location='cpu',weights_only=False,mmap=True)
    epoch = saved['epoch']
    from handoff import audited_event
    audited_event(PARENT,epoch)
    net = build('unet_mean')
    assert sum(v.numel() for v in net.parameters()) == 32142859
    opt = torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-6)
    restore(saved,net,opt,scheduler,cuda=False)
    assert {int(v['step']) for v in opt.state.values()} == {saved['updates']}
    steps = saved['updates']
    del saved,net,opt,scheduler
    gc.collect()
    # Precompute only the first extension epoch; subsequent features roll ahead.
    prepare_features(ROLES[0],51)
    dataset = Mixtures(ROLES[0],51)
    for k in (1,2,3):
        i = int(np.flatnonzero(dataset.rows['count']==k)[0])
        fast,full = dataset[i],dataset.full_example(i)
        for key in ('mixture','references','context_features'):
            np.testing.assert_allclose(fast[key],full[key],rtol=2e-6,atol=2e-7)
    report = dict(status='PASS', all_extension_schedules_checked=50,
        first_epoch=51,total_epochs=100,examples_per_epoch=2400,
        model_parameters=32142859,parent_model_optimizer_scheduler_cpu_roundtrip_exact=True,
        preflight_parent_epoch=epoch,preflight_optimizer_updates=steps,
        actual_epoch50_handoff_pending=True, unchanged_data_loader_except_run_path=True,
        fixed_dev_schedule_sha256=parent['fixed_dev_schedule_sha256'],
        fixed_dev_feature_bytes_identical=True,long_short_equivalence='PASS',
        parent_source_integrity='PASS', heldout_read=False, independent_recording_validation=False,
        preparation_sha256=sha256(PREP/'PREPARATION.json'),
        source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in HERE.glob('*.py')})
    write_json(OUT/'READY.json',report)
    write_json(PUBLIC/'TIMEBLOCK100_PREPARATION.json',report)
    print('EXTENSION_PREPARATION_PASS',flush=True)


if __name__ == '__main__':
    main()
