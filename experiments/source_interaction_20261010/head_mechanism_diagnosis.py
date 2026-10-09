"""Read-only TRAIN4 diagnosis of the e2 head, including a cross-term ledger.

Measures internal contrast on a deterministic grid of TF points. Component
readouts are additive algebra, not independently trained causal effects.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import train_comparison as worker

w = worker.watch


def centered(x):
    return x - x.mean(1, keepdim=True)


def energy(x):
    return float(x.square().sum())


@torch.inference_mode()
def measure(head, features):
    raw = head.base(features)
    flat = features.permute(0, 2, 3, 1).reshape(-1, 64)
    sources = raw[:, :6].permute(0, 2, 3, 1).reshape(-1, 3, 2)
    ix = torch.linspace(0, len(flat)-1, 4096).long().unique()
    f, r = flat[ix].double(), sources[ix].double()
    precise = copy.deepcopy(head).double()
    h = precise.input(torch.cat((f[:, None].expand(-1, 3, -1), r), -1))
    q, k, v = precise.qkv(precise.norm(h)).chunk(3, -1)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head.width), -1)
    attention = precise.attention_output(a @ v)
    ff = precise.feedforward(h + attention)
    parts = [centered(precise.readout(x)) for x in (h, attention, ff)]
    delta = sum(parts)
    expected = precise.correction(f, r)
    torch.testing.assert_close(delta, expected, rtol=1e-9, atol=1e-12)
    pair_terms = [dict(i=i, j=j, twice_inner_product=float(2 * (parts[i]*parts[j]).sum()))
                  for i in range(3) for j in range(i+1, 3)]
    reconstructed = sum(energy(x) for x in parts) + sum(x['twice_inner_product'] for x in pair_terms)
    np.testing.assert_allclose(reconstructed, energy(delta), rtol=1e-9, atol=1e-15)
    return dict(points=len(ix), contrast_definition='Subtract mean over the three raw source slots at every TF point',
        raw_source_energy=energy(r), raw_source_contrast_energy=energy(centered(r)),
        input_latent_energy=energy(h), input_latent_contrast_energy=energy(centered(h)),
        attention_energy=energy(attention), attention_contrast_energy=energy(centered(attention)),
        attention_weight_row_contrast_energy=energy(centered(a)),
        feedforward_contrast_energy=energy(centered(ff)),
        readout_component_names=['input_latent', 'attention', 'feedforward'],
        readout_component_energies=[energy(x) for x in parts], readout_cross_terms=pair_terms,
        final_correction_energy=energy(delta), decomposition_pass=True,
        measurement_dtype='float64 copy of trained FP32 head; backbone features from FP32 CPU')


@torch.inference_mode()
def run(study, root, public):
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate mechanism diagnosis')
    p = w.read(study / 'PROTOCOL.json')
    pre = w.read(Path(p['dependency']) / 'PROTOCOL.json')
    checkpoint = study / 'source_interaction/ACTUAL_002.pt'
    receipt = w.read(study / 'source_interaction/EPOCH_002.json')
    if receipt['updates'] != 150 or receipt['protocol_sha256'] != w.digest(study/'PROTOCOL.json'):
        raise ValueError('Completed candidate e2 required')
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__).relative_to(w.ROOT))] = w.digest(Path(__file__))
    protocol = dict(status='REGISTERED_CPU_TRAIN4_HEAD_MECHANISM', source_sha256=sources,
        study_protocol_sha256=w.digest(study/'PROTOCOL.json'), checkpoint_sha256=w.digest(checkpoint),
        train_indices=pre['train_indices'],
        selection='Same four preflight TRAIN rows; first-to-last equally spaced 4096 TF indices per row',
        model='Full candidate actual e2, 150 updates, no weight changes',
        stage_analysis='Shared-feature input / source attention / feedforward additive readout; all cross terms retained',
        model_updates=0, gpu_use=False, heldout_read=False, validation_read=False, registered_at=time.time())
    w.write(root / 'PROTOCOL.json', protocol)
    for rel, sha in sources.items():
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((w.ROOT/rel).read_bytes())
        if w.digest(target) != sha:
            raise ValueError('Frozen source changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['epoch'] != 2 or saved['updates'] != 150 or saved['protocol_sha256'] != protocol['study_protocol_sha256']:
        raise ValueError('Wrong checkpoint')
    net = worker.make_model('source_interaction').eval()
    net.load_state_dict(saved['model'])
    del saved
    train = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    rows = []
    for index in protocol['train_indices']:
        item = train[index]
        batch = {k: torch.as_tensor(np.asarray(item[k])[None]) for k in
            ('mixture', 'references', 'active', 'context_features', 'crop_start')}
        captured = []
        handle = net.output.register_forward_pre_hook(lambda head, args: captured.append(measure(head, args[0])))
        estimates, logits = worker.predict(net, batch)
        handle.remove()
        if len(captured) != 1 or not torch.isfinite(estimates).all():
            raise ValueError('Unexpected full forward')
        rows.append(dict(index=index, count=item['construction_count'], **captured[0]))
        w.write(root/'STATE.json', dict(status='CPU_TRAIN4_HEAD_MECHANISM', cases=len(rows), total=4,
            pid=os.getpid(), time=time.time()))
    for rel, sha in sources.items():
        if w.digest(w.ROOT/rel) != sha:
            raise ValueError('Frozen source changed')
    if w.digest(checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint modified')
    result = dict(status='COMPLETE', protocol_sha256=w.digest(root/'PROTOCOL.json'),
        checkpoint_sha256=protocol['checkpoint_sha256'], rows=rows, checks='Exact centered readout decomposition including cross terms, all four forwards finite',
        model_updates=0, heldout_read=False, validation_read=False,
        limitation='TRAIN4 sampled TF points, feature-scale-dependent descriptive internal measurements; no isolated causal contribution or generalization estimate')
    w.write(root/'COMPLETE.json', result)
    w.write(public, result)
    w.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(dict(status='COMPLETE', cases=len(rows)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study','run','public'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.study.resolve(), args.run.resolve(), args.public.resolve())
    except Exception:
        w.write(args.run/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
