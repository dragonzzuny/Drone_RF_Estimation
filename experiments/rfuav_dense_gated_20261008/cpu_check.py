"""One real TRAINING mixture, full native window, full-size forward on CPU.

This is implementation validation, NOT training or a performance experiment.
No validation/held-out I/Q is read. Output contains geometry/timing, no waveform.
"""
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from models import ARMS, build, predict
from study import DEFAULT_PREPARATION, admitted_dataset, batch
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256


if __name__ == '__main__':
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[-2:]))
    os.nice(10)
    torch.set_num_threads(2)
    dataset = admitted_dataset(DEFAULT_PREPARATION, 'train_pack', 1)
    index = int(np.flatnonzero(dataset.rows['count'] == 3)[0])
    item = batch(dataset[index], 'cpu')
    receipts = []
    for arm in ARMS:
        started = time.time()
        net = build(arm).eval()
        with torch.no_grad():
            estimates, logits = predict(net, item)
        error = float((estimates.sum(1) - item['mixture']).abs().square().mean()
                      / item['mixture'].abs().square().mean())
        assert torch.isfinite(estimates).all() and torch.isfinite(logits).all()
        assert estimates.shape == (1, 4, 63872) and error < 1e-10
        entry = dict(arm=arm, parameters=sum(p.numel() for p in net.parameters()),
            output_shape=list(estimates.shape), count_shape=list(logits.shape),
            sum_relative_error=error, elapsed_seconds=time.time() - started)
        receipts.append(entry)
        print(entry, flush=True)
        del net, estimates, logits
        gc.collect()
    output = Path(__file__).resolve().parent / 'CPU_FULL_INPUT_CHECK.json'
    write_json(output, dict(status='PASS_FULL_CAPACITY_NATIVE_INPUT_FORWARD_CPU', time=time.time(),
        source_role='train_pack', example_index=index, construction_count=3,
        preparation_sha256=sha256(DEFAULT_PREPARATION / 'PREPARATION.json'),
        input_samples=63872, fs_hz=100000000, arms=receipts,
        trained=False, gpu_preflight_passed=False, reconstruction_quality_evaluated=False,
        validation_iq_read=False, heldout_iq_read=False))
