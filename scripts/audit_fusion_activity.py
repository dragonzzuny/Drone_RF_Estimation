"""CPU-only, train-only functional diagnostics on an immutable milestone.

Inference removals show sensitivity, not the causal benefit of retraining an
ablated architecture. They never select a checkpoint or change GPU training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/rfuav_local_global_20261008'))
from fusion_models import build, predict, stack_batch
from long_data import admitted_dataset
from drone_rf.waveform import waveform_metrics


@torch.no_grad()
def diagnose(run, epoch):
    torch.set_num_threads(2)
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    preparation = next(Path(p).parent for p in protocol['data_sha256'] if p.endswith('/PREPARATION.json'))
    data = admitted_dataset(preparation, 'train_pack', 1)
    indices = [int(i) for k in (2, 3) for i in np.flatnonzero(data.rows['count'] == k)[:2]]
    net = build('local_global').eval()
    initial = {k: v.clone() for k, v in net.state_dict().items() if k.startswith(('iq_context.', 'fine_film.', 'coarse_film.'))}
    checkpoint = run/'local_global'/f'SELECTED_{epoch:03d}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protocol_sha256 = hashlib.sha256((run/'PROTOCOL.json').read_bytes()).hexdigest()
    if saved['protocol_sha256'] != protocol_sha256:
        raise ValueError('Checkpoint does not match the run protocol')
    net.load_state_dict(saved['model'])
    changes = {}
    for prefix in ('iq_context.', 'fine_film.', 'coarse_film.'):
        reference = difference = 0.
        for name, before in initial.items():
            if name.startswith(prefix):
                reference += float(before.double().square().sum())
                difference += float((net.state_dict()[name] - before).double().square().sum())
        changes[prefix] = dict(relative_parameter_l2=(difference/reference)**.5)
    rows = []
    for index in indices:
        item = stack_batch([data[index]], 'cpu')
        full, _ = predict(net, item)
        row = dict(index=index, construction_count=int(item['construction_count'][0]), modes={})
        for mode in ('full', 'fusion_bypass', 'without_fine', 'without_coarse'):
            handle = None
            if mode == 'full':
                estimate = full
            else:
                if mode == 'fusion_bypass':
                    net.fusion_enabled = False
                elif mode == 'without_fine':
                    handle = net.fine_film.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
                else:
                    handle = net.coarse_film.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
                try:
                    estimate, _ = predict(net, item)
                finally:
                    net.fusion_enabled = True
                    if handle is not None:
                        handle.remove()
            active = item['active'][0]
            metric = waveform_metrics(estimate, item['references'], item['active'], item['mixture'])
            row['modes'][mode] = dict(mean_nmse=float(metric['nmse'][0][active].mean()),
                output_change_relative_energy=float((estimate-full).abs().square().sum()/full.abs().square().sum()))
        rows.append(row)
    return dict(status='TRAIN_ONLY_CPU_DIAGNOSTIC_COMPLETE',selected_epoch=saved['best']['epoch'],
        milestone_epoch=epoch,protocol_sha256=protocol_sha256,
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model='local_global',examples=4,parameter_changes=changes,rows=rows,
        inference_removals_not_matched_retraining=True,validation_or_heldout_iq_read=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--epoch',type=int,choices=(1,5),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.nice(10)
    cpus=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(cpus[-4:-2] or cpus))
    start=time.time(); result=diagnose(args.run,args.epoch); result['seconds']=time.time()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
