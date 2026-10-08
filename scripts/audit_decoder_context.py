"""Train-only CPU checks of learned context use and global phase consistency.

Reference-free alignment compares predictions with predictions. References are
used only to score the results. Inference perturbations are not retraining
ablations and four examples cannot establish generalization.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/rfuav_decoder_fusion_20261009'))
from late_models import build, predict, stack_batch
from long_data import admitted_dataset
from drone_rf.waveform import waveform_metrics


def align_predictions(predicted, anchor):
    """One permutation per whole window for three slots; fixed background.

    No target waveforms, count labels, identities, phase/gain fitting, or
    per-sample permutation switching are used here.
    """
    if predicted.shape != anchor.shape or predicted.shape[:2] != (1, 4):
        raise ValueError('Expected one four-output prediction')
    perms = list(itertools.permutations(range(3)))
    costs = torch.stack([(predicted[:, p] - anchor[:, :3]).abs().square().mean() for p in perms])
    order = perms[int(costs.argmin())]
    return torch.cat((predicted[:, order], predicted[:, 3:]), 1), list(order)


@torch.no_grad()
def diagnose(run, milestone):
    torch.set_num_threads(2)
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    digest = hashlib.sha256((run/'PROTOCOL.json').read_bytes()).hexdigest()
    checkpoint = run/'local_global'/f'SELECTED_{milestone:03d}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['protocol_sha256'] != digest:
        raise ValueError('Checkpoint protocol mismatch')
    net = build('local_global').eval(); net.load_state_dict(saved['model'])
    preparation = next(Path(p).parent for p in protocol['data_sha256'] if p.endswith('/PREPARATION.json'))
    data = admitted_dataset(preparation, 'train_pack', 1)
    indices = [int(i) for k in (2, 3) for i in np.flatnonzero(data.rows['count'] == k)[:2]]
    rows = []
    for index in indices:
        item = stack_batch([data[index]], 'cpu')
        full, _ = predict(net, item)

        def score(estimate):
            metric = waveform_metrics(estimate, item['references'], item['active'], item['mixture'])
            mask = item['active'][0]
            return dict(mean_nmse=float(metric['nmse'][0][mask].mean()),
                mean_si_sdr=float(metric['si_sdr'][0][mask].mean()),
                sum_relative_error=float(metric['sum_relative_error'][0]),
                output_change_relative_energy=float((estimate-full).abs().square().sum()/full.abs().square().sum()))

        row = dict(index=index, count=int(item['construction_count'][0]), modes={'full':score(full)})
        for mode in ('fusion_bypass', 'zero_context_features', 'reverse_context_tokens'):
            hook = None
            if mode == 'fusion_bypass':
                net.fusion_enabled = False
            elif mode == 'zero_context_features':
                hook = net.iq_context.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
            else:
                hook = net.iq_context.register_forward_hook(lambda m, i, o: o.flip(-1))
            try:
                estimate, _ = predict(net, item)
            finally:
                net.fusion_enabled = True
                if hook is not None:
                    hook.remove()
            estimate, order = align_predictions(estimate, full)
            row['modes'][mode] = dict(score(estimate), prediction_only_order=order)
        estimates = [full]
        for degrees, factor in ((90, 1j), (180, -1+0j), (270, -1j)):
            changed = dict(item, mixture=item['mixture']*factor, long_mixture=item['long_mixture']*factor)
            estimate, _ = predict(net, changed)
            estimate, order = align_predictions(estimate/factor, full)
            row['modes'][f'global_phase_{degrees}'] = dict(score(estimate), prediction_only_order=order)
            estimates.append(estimate)
        row['modes']['four_phase_average'] = score(torch.stack(estimates).mean(0))
        rows.append(row)
        print(json.dumps(dict(index=index, complete=len(rows), total=4)), flush=True)
    return dict(status='TRAIN_ONLY_CPU_COMPLETE', protocol_sha256=digest,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        milestone_epoch=milestone, selected_epoch=saved['best']['epoch'], rows=rows,
        prediction_alignment='three source slots aligned to original prediction by whole-window unscaled L2; background fixed',
        model_updates=0, inference_labels_used=False, heldout_or_validation_iq_read=False,
        interpretation='four train cases; altered inference, not architecture retraining or generalization evidence')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--milestone',type=int,choices=(1,5),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.nice(10)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2] or cpus))
    start=time.time(); result=diagnose(args.run,args.milestone); result['seconds']=time.time()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
