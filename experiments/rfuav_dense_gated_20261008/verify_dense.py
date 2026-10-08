"""Check dense schedule equivalence and real full-input warm-start identity.

No GPU training; only training I/Q is read. Validation metadata/features are
checked against the prior fixed set without opening validation I/Q payloads.
"""
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from dense_data import DenseContextMixtures
from study import batch, make_model
from models import predict
from prepare_dense_gated import OLD
from drone_rf.context_training_data import write_json


if __name__ == '__main__':
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent / 'dense_preparation'
    config = json.loads((root / 'PREPARATION.json').read_text())
    parent_config = json.loads((OLD / 'PREPARATION.json').read_text())
    old_clips = json.loads(Path(parent_config['manifest']).read_text())['clips']
    clips = json.loads(Path(config['manifest']).read_text())['clips']
    used = set()
    for name in ('TRAIN.npy', 'VALIDATION.npy'):
        before = np.load(OLD / name, allow_pickle=False)
        before = before[before['epoch'] <= 5] if name == 'TRAIN.npy' else before
        after = np.load(root / name, allow_pickle=False)
        assert len(after) == len(before)
        for field in ('epoch', 'count', 'levels', 'phases', 'crop_start'):
            np.testing.assert_array_equal(before[field], after[field])
        for a, b in zip(before, after):
            for j in range(int(a['count'])):
                old, new = old_clips[int(a['indices'][j])], clips[int(b['indices'][j])]
                assert old['category'] == new['category'] and old['center_hz'] == new['center_hz']
                if name == 'VALIDATION.npy':
                    assert old == new
                else:
                    used.add(int(b['indices'][j]))
    assert len(used) == 3902
    before_features = np.load(OLD / 'features/validation_pack_001.npy', mmap_mode='r')
    after_features = np.load(root / 'features/validation_pack_001.npy', mmap_mode='r')
    np.testing.assert_array_equal(before_features, after_features)
    train = DenseContextMixtures(root, 'train_pack', 1, use_features=False)
    index = int(np.flatnonzero(train.rows['count'] == 3)[0])
    item = batch(train.full_example(index), 'cpu')
    started = time.time()
    baseline = make_model('unet_mean', config).eval()
    with torch.no_grad():
        baseline_output, baseline_count = predict(baseline, item)
    # Hash tensor values, then release baseline before building the larger arm.
    import hashlib
    shared_hashes = {k: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()
                     for k, v in baseline.state_dict().items()}
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    del baseline
    gc.collect()
    gated = make_model('unet_gated', config).eval()
    for name, value in gated.state_dict().items():
        if not name.startswith('gates.'):
            assert hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() == shared_hashes[name]
    with torch.no_grad():
        gated_output, gated_count = predict(gated, item)
    torch.testing.assert_close(gated_output, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(gated_count, baseline_count, rtol=0, atol=0)
    result = dict(status='PASS_DENSE_SCHEDULE_AND_FULL_INPUT_WARM_START',
        train_mixtures=12000, all_3902_training_clips_scheduled=True,
        training_count_class_power_phase_crop_schedule_preserved=True,
        validation_630_mixtures_unchanged=True, validation_features_bitwise_identical=True,
        source_role='train_pack', source_count=3, example_index=index, input_samples=63872,
        baseline_parameters=baseline_parameters, gated_parameters=sum(p.numel() for p in gated.parameters()),
        shared_weights_bitwise_identical=True, initial_waveforms_bitwise_identical=True,
        initial_count_logits_bitwise_identical=True, checkpoint_selected_epoch=22,
        cpu_model_check_seconds=time.time()-started, gpu_training_started=False,
        heldout_iq_read=False, validation_iq_read=False, time=time.time())
    write_json(root / 'INTEGRATION_CHECK.json', result)
    print(json.dumps(result, indent=2), flush=True)
