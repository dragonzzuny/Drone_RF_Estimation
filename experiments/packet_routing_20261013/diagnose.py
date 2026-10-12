"""Read-only stage diagnosis of the already exposed epoch79 FPV/Mavic event.

Reference fits below are diagnostics, never prediction-time slot alignment.
No source waveform, packet payload, serial or position is written to reports.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / 'uav_analysis/rf_detection/separator_detector_20261012'
sys.path.insert(0, str(ROOT / 'experiments/bandwidth_timeblock_20261011'))
from data import Mixtures  # Register the pinned model/vendor import paths.
from models import build, predict
sys.path.insert(0, str(ROOT / 'experiments/phase_eight_20261011'))
from core import four
from drone_rf.context_data import mixture_context_features, component_gains
from drone_rf.waveform import analyze, synthesize, waveform_metrics

PERMS = list(itertools.permutations(range(3)))


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def routing(y, refs, block=8192):
    """Joint complex LS; squared coefficients are NOT additive power fractions."""
    rows = []
    for lo in range(0, refs.shape[-1], block):
        hi = min(lo + block, refs.shape[-1])
        r = refs[:, lo:hi].astype(np.complex128)
        out = y[:, lo:hi].astype(np.complex128)
        energy = np.sum(abs(r)**2, axis=1)
        if np.any(energy <= 0):
            continue
        scale = np.sqrt(energy)
        design = (r / scale[:, None]).T
        gram = design.conj().T @ design
        coefficients = np.linalg.lstsq(design, out.T, rcond=None)[0] / scale[:, None]
        explained = r.T @ coefficients
        rows.append(dict(lo=lo, hi=hi, reference_power=(energy/(hi-lo)).tolist(),
            gram_condition=float(np.linalg.cond(gram)),
            coefficient_abs2=(abs(coefficients)**2).tolist(),
            dominant_source_slot=np.argmax(abs(coefficients[:, :3])**2, axis=1).tolist(),
            unexplained_output_power=np.mean(abs(out.T-explained)**2, axis=0).tolist()))
    return rows


def metrics(y, refs, mixture):
    target = np.zeros((3, refs.shape[-1]), np.complex64)
    target[:2] = refs
    m = waveform_metrics(torch.from_numpy(y.astype(np.complex64))[None],
        torch.from_numpy(target)[None], torch.tensor([[True, True, False]]),
        torch.from_numpy(mixture.astype(np.complex64))[None])
    return dict(assignment=m['assignment'][0].tolist(), nmse=m['nmse'][0, :2].tolist(),
        si_sdr=m['si_sdr'][0, :2].tolist(), sum_relative_error=float(m['sum_relative_error'][0]))


def run(out, checkpoint):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        raise RuntimeError('Use a new output directory; do not overwrite a registered run')
    plan = read(OLD/'raw_diagnosis/SEPARATION_PLAN.json')
    e79 = read(OLD/'retest_best_e079/PLAN.json')
    inputs = [Path(__file__), checkpoint, OLD/'raw_diagnosis/SEPARATION_PLAN.json',
        OLD/'raw_diagnosis/FIRST_EVENT.npz', Path(plan['second_clip']['cache_path']),
        OLD/'retest_best_e079/SEPARATED_EVENT.npz']
    source = [ROOT/'experiments/rfuav_dense_gated_20261008/models.py',
              ROOT/'experiments/phase_eight_20261011/core.py',
              ROOT/'experiments/rfuav_phase_average_20261009/phase_core.py']
    source += sorted((ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf').glob('*.py'))
    hashes = {str(p): sha(p) for p in inputs + source}
    if checkpoint == Path(e79['checkpoint']):
        assert hashes[str(checkpoint)] == e79['checkpoint_sha256']
    assert hashes[str(inputs[3])] == plan['first_event_hash']
    assert hashes[str(inputs[4])] == plan['second_clip']['cache_sha256']
    write(out/'PLAN.json', dict(time=time.time(), checkpoint=str(checkpoint), hashes=hashes,
        purpose='stage diagnosis on previously exposed DEV event; no training or checkpoint selection',
        stages=['zero_phase_raw_unprojected', 'zero_phase', 'C4', 'C8'],
        merge=['Hann floor .001', 'uniform'], references_in_inference=False,
        fitting='joint complex least squares, diagnostic only', fs_hz=100000000,
        caveat='Raw common mode is not identifiable under equal mixture projection; pre/post is not a causal ablation.'))
    first = np.load(inputs[3])['filtered']
    second = np.load(inputs[4])
    gains = component_gains([float(np.mean(abs(x.astype(np.complex128))**2)) for x in (first, second)], [0,0], [0,.7])
    # NumPy complex128 scalar promotion must match the original synthesis;
    # Python complex scalars round differently with NumPy complex64 arrays.
    archived = np.array([complex(*g) for g in plan['gains_real_imag']], dtype=np.complex128)
    assert np.array_equal(gains, archived)
    refs = np.stack([(x*g).astype(np.complex64) for x,g in zip((first,second), gains)])
    mixture = refs.sum(0)
    features = mixture_context_features(mixture)['context_features']
    lo, hi = plan['roi']
    prior = np.load(OLD/'retest_best_e079/SEPARATED_EVENT.npz')
    assert np.array_equal(prior['references'][:2], refs[:,lo:hi])
    assert np.array_equal(prior['mixture'], mixture[lo:hi])
    assert np.array_equal(prior['context_features'], features)
    write(out/'INPUT_VERIFICATION.json', dict(status='PASS', exact=['references','mixture','context_features']))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    net = build('unet_mean').eval().requires_grad_(False)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model'])
    epoch = ckpt['epoch']
    del ckpt
    captured = []
    hook = net.output.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    starts, length = plan['starts'], plan['window_samples']
    start0, end = starts[0], starts[-1]+length
    windows = {s: [] for s in ('raw','zero','C4','C8')}
    details = []
    for j, start in enumerate(starts):
        begin = time.monotonic()
        item = dict(mixture=torch.from_numpy(mixture[start:start+length].copy())[None],
            context_features=torch.from_numpy(features.copy())[None], crop_start=torch.tensor([start]))
        views, raw = [], None
        orders = []
        with torch.inference_mode():
            for degree in (0,90,180,270,45,135,225,315):
                factor = {0:1+0j,90:1j,180:-1+0j,270:-1j}.get(degree)
                if factor is None:
                    factor = complex(math.cos(math.radians(degree)), math.sin(math.radians(degree)))
                view = dict(item, mixture=item['mixture']*factor)
                captured.clear()
                pred, count = predict(net, view)
                if degree == 0:
                    z = analyze(item['mixture'])
                    scale = z.abs().square().mean((1,2), keepdim=True).sqrt().clamp_min(1e-8)
                    r = captured[0].float()
                    raw = synthesize(torch.complex(r[:,0::2], r[:,1::2])*scale[:,None], length)[0].numpy()
                    anchor = pred
                    aligned = pred
                    order = [0,1,2]
                else:
                    aligned, order = four.align_predictions(pred/factor, anchor)
                views.append(aligned)
                orders.append(order)
            mean4 = torch.stack(views[:4]).mean(0)
            mean8 = .5*(mean4+torch.stack(views[4:]).mean(0))
        stage = dict(raw=raw, zero=views[0][0].numpy(), C4=mean4[0].numpy(), C8=mean8[0].numpy())
        rows = {}
        for name, y in stage.items():
            windows[name].append(y)
            rows[name] = dict(metrics=metrics(y,refs[:,start:start+length],mixture[start:start+length]),
                             blocks=routing(y,refs[:,start:start+length]))
        details.append(dict(index=j,start=start,phase_orders=orders,stage=rows,seconds=time.monotonic()-begin))
        write(out/'WINDOWS.json', details)
        write(out/'STATE.json',dict(status='DIAGNOSING',epoch=epoch,window=j+1,windows=len(starts),pid=os.getpid()))
        print('WINDOW',j+1,'/',len(starts),'seconds',round(time.monotonic()-begin,1),flush=True)
    hook.remove()
    assembled = {}
    results = {}
    for name, ys in windows.items():
        for weight_name in ('hann','uniform'):
            total = np.zeros((4,end-start0), np.complex128)
            weights = np.zeros(end-start0)
            win = np.maximum(np.hanning(length),.001) if weight_name=='hann' else np.ones(length)
            receipts=[]
            for start,y0 in zip(starts,ys):
                local = start-start0
                ov = weights[local:local+length]>0
                costs, order=[],(0,1,2)
                if ov.any():
                    anchor = total[:3,local:local+length][:,ov]/weights[local:local+length][ov]
                    costs=[float(np.mean(abs(y0[list(p)][:,ov]-anchor)**2)) for p in PERMS]
                    order=PERMS[int(np.argmin(costs))]
                y=y0[[*order,3]]
                total[:,local:local+length]+=y*win
                weights[local:local+length]+=win
                receipts.append(dict(start=start,order=list(order),costs=costs))
            y=(total[:,lo-start0:hi-start0]/weights[lo-start0:hi-start0]).astype(np.complex64)
            key=name+'_'+weight_name
            assembled[key]=y
            results[key]=dict(metrics=metrics(y,refs[:,lo:hi],mixture[lo:hi]),
                routing=routing(y,refs[:,lo:hi],25000),overlap_matching=receipts)
    difference=float(np.max(abs(assembled['C8_hann']-prior['estimates'])))
    if epoch == 79:
        assert np.allclose(assembled['C8_hann'],prior['estimates'],rtol=2e-5,atol=2e-6), difference
    # Compact 8-window C8/zero predictions allow alternate assembly without 64 new forwards.
    np.savez_compressed(out/'WINDOW_PREDICTIONS.npz', starts=starts,
        zero=np.stack(windows['zero']), C8=np.stack(windows['C8']))
    np.savez_compressed(out/'ASSEMBLED.npz', **assembled)
    result=dict(status='COMPLETE',epoch=epoch,original_e79_max_difference=difference,stages=results,
        reference_fit_is_oracle_diagnostic=True,independent_recordings=False)
    write(out/'RESULT.json',result)
    write(out/'STATE.json',dict(status='COMPLETE',epoch=epoch,pid=os.getpid(),windows=len(starts)))
    write(out/'VERIFICATION.json',dict(status='PASS',input_hashes_unchanged=all(sha(p)==v for p,v in hashes.items()),
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='VERIFICATION.json'}))
    print('COMPLETE',epoch,'C8_HANN',results['C8_hann']['metrics'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.checkpoint)
