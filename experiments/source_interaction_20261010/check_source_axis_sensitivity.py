"""Synthetic local sensitivity of the source-axis attention near equal slots.

No trained model or RF samples. This checks a symmetry statement, not the
frequency of equal slots in RF data and not an explanation of all residuals.
"""
import json
import math
from pathlib import Path
import hashlib
import torch
from torch import nn
from source_head import SourceInteractionHead


def run():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    head = SourceInteractionHead(nn.Conv2d(64, 8, 1), checkpoint_chunks=False).double()
    with torch.no_grad():
        nn.init.normal_(head.readout.weight, std=.01)
    f = torch.randn(128, 64, dtype=torch.float64)
    common = torch.randn(128, 1, 2, dtype=torch.float64)
    direction = torch.randn(128, 3, 2, dtype=torch.float64)
    direction -= direction.mean(1, keepdim=True)
    rows = []
    def centered_rms(x):
        return float((x - x.mean(1, keepdim=True)).square().mean().sqrt())
    with torch.inference_mode():
        equal = common.expand(-1, 3, -1)
        same = head.correction(f, equal)
        if float(same.abs().max()) > 1e-14:
            raise ValueError('Equal-slot symmetry failed')
        for eps in (1., .5, .25, .125, .0625, .03125):
            raw = common + eps * direction
            h = head.input(torch.cat((f[:, None].expand(-1, 3, -1), raw), -1))
            q, k, v = head.qkv(head.norm(h)).chunk(3, -1)
            weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head.width), -1)
            attention = head.attention_output(weights @ v)
            rows.append(dict(epsilon=eps, input_source_contrast=centered_rms(h),
                attention_source_contrast=centered_rms(attention),
                whole_head_correction_contrast=centered_rms(head.correction(f, raw))))
    orders = []
    for a, b in zip(rows, rows[1:]):
        orders.append(dict(epsilon=b['epsilon'],
            input_order=math.log2(a['input_source_contrast'] / b['input_source_contrast']),
            attention_order=math.log2(a['attention_source_contrast'] / b['attention_source_contrast']),
            whole_head_order=math.log2(a['whole_head_correction_contrast'] / b['whole_head_correction_contrast'])))
    if not (abs(orders[-1]['input_order'] - 1) < .01 and abs(orders[-1]['attention_order'] - 3) < .1
            and abs(orders[-1]['whole_head_order'] - 1) < .1):
        raise ValueError('Synthetic local scaling differs from derivation')
    result = dict(status='PASS', checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_head_sha256=hashlib.sha256(Path(__file__).with_name('source_head.py').read_bytes()).hexdigest(),
        rows=rows, halving_orders=orders, equal_slot_max_correction=float(same.abs().max()),
        rf_reads=0, checkpoint_reads=0, gpu_use=False, optimizer_updates=0,
        synthetic_nonzero_readout=True,
        interpretation='Near three equal raw slots, centered attention output is cubic in their contrast; the full head retains a linear residual/MLP path. No claim that actual RF slots lie in this regime.')
    path = Path(__file__).resolve().parents[2] / 'reports/2026-10-10/SOURCE_AXIS_SENSITIVITY.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
