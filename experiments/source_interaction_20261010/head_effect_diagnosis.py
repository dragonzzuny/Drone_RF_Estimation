"""TRAIN4 inference-only added-head ablation; never edits saved weights."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import train_comparison as worker
from drone_rf.waveform import waveform_metrics

w = worker.watch


@torch.inference_mode()
def run(study, dependency, root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate head-effect diagnosis')
    p = w.read(study / 'PROTOCOL.json')
    preflight = w.read(Path(p['dependency']) / 'PROTOCOL.json')
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__).relative_to(w.ROOT))] = w.digest(Path(__file__))
    indices = preflight['train_indices']
    protocol = dict(status='REGISTERED_CPU_TRAIN4_HEAD_EFFECT', source_sha256=sources,
        study_protocol_sha256=w.digest(study / 'PROTOCOL.json'),
        checkpoint='source_interaction ACTUAL_001, entire adapted backbone held fixed',
        indices=indices, selection='same four preflight TRAIN mixtures; not chosen by ablation results',
        modes=['trained_head_on', 'trained_head_off'],
        intervention='remove only the extra output branch of a loaded CPU copy; preserve its trained base head',
        training_updates=0, heldout_read=False, validation_reads=0,
        inference_reference_access=False, registered_at=time.time())
    w.write(root / 'PROTOCOL.json', protocol)
    while not (dependency / 'COMPLETE.json').exists():
        if (dependency / 'FAILURE.json').exists():
            raise RuntimeError('Prior CPU oracle diagnosis failed')
        if time.time() - protocol['registered_at'] > 7200:
            raise TimeoutError('CPU dependency did not finish')
        w.write(root / 'STATE.json', dict(status='WAITING_CPU_ORACLE', pid=os.getpid(), time=time.time()))
        time.sleep(30)
    checkpoint = study / 'source_interaction/ACTUAL_001.pt'
    before = w.digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['epoch'] != 1 or saved['updates'] != 75 or saved['protocol_sha256'] != protocol['study_protocol_sha256']:
        raise ValueError('Wrong checkpoint')
    net = worker.make_model('source_interaction').eval()
    net.load_state_dict(saved['model'])
    del saved
    head = net.output
    data = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    rows = []
    for index in indices:
        item = data[index]
        batch = {k: torch.as_tensor(np.asarray(item[k])[None]) for k in
                 ('mixture', 'references', 'active', 'context_features', 'crop_start')}
        estimates, logits = {}, {}
        values = {}
        for mode in protocol['modes']:
            net.output = head if mode == 'trained_head_on' else head.base
            estimate, count_logits = worker.predict(net, batch)
            score = waveform_metrics(estimate, batch['references'], batch['active'], batch['mixture'])
            n = item['construction_count']
            estimates[mode] = estimate.to(torch.complex128)
            logits[mode] = count_logits
            values[mode] = dict(nmse=score['nmse'][0, :n].tolist(), si_sdr=score['si_sdr'][0, :n].tolist(),
                assignment=score['assignment'][0].tolist(), sum_relative_error=float(score['sum_relative_error'].max()))
            if values[mode]['sum_relative_error'] > 1e-9 or not np.isfinite(values[mode]['nmse'] + values[mode]['si_sdr']).all():
                raise ValueError('Invalid metric')
        torch.testing.assert_close(logits['trained_head_on'], logits['trained_head_off'], rtol=0, atol=0)
        delta = estimates['trained_head_on'] - estimates['trained_head_off']
        mix_power = batch['mixture'].to(torch.complex128).abs().square().mean()
        effect = delta.abs().square().mean(-1)[0] / mix_power
        rows.append(dict(index=index, count=item['construction_count'], metrics=values,
            raw_slot_delta_energy_over_mixture=effect.tolist(), count_logits_identical=True,
            note='Per-reference waveform metrics independently PIT aligned; delta energies retain raw slot identities'))
        net.output = head
        w.write(root / 'STATE.json', dict(status='CPU_TRAIN4_HEAD_EFFECT', cases=len(rows), total=4,
            pid=os.getpid(), time=time.time()))
    if len(rows) != 4 or w.digest(checkpoint) != before:
        raise ValueError('Incomplete diagnosis or modified checkpoint')
    for rel, sha in sources.items():
        if w.digest(w.ROOT / rel) != sha:
            raise ValueError('Frozen source changed')
    result = dict(status='COMPLETE', protocol_sha256=w.digest(root / 'PROTOCOL.json'),
        checkpoint_sha256=before, rows=rows, training_updates=0, heldout_read=False,
        interpretation='Direct e1 head contribution on TRAIN4 with identical adapted backbone; not generalization or full-training causal attribution')
    w.write(root / 'COMPLETE.json', result)
    w.write(public, result)
    w.write(root / 'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'dependency', 'run', 'public'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.study.resolve(), args.dependency.resolve(), args.run.resolve(), args.public.resolve())
    except Exception:
        w.write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
