"""Full-capacity joint-context feasibility, not a scientific training result.

Use one admitted FIT three-source sequence; discard the single update. Do not
modify registered trainers, access held-out data, or save new model weights.
"""
from pathlib import Path
import fcntl
import gc
import hashlib
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/packet_routing_20261013'))
from sequence_data import Sequences, ROLES
from train_chunk_candidate import load, step_bundle, LOCK
from chunk_temporal_candidate import predict_bundle
from models import build, predict
from study import batch
import torch

OUT = ROOT / 'reports/2026-10-13/JOINT_CONTEXT_FEASIBILITY.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    assert not OUT.exists(), 'Do not replace an executed receipt'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = ROOT / 'local/timeblock_extend100_20261012_v1/ACTUAL_099.pt'
    expected = json.loads(checkpoint.with_name('EPOCH_099.json').read_text())['checkpoint_sha256']
    assert sha(checkpoint) == expected
    receipt = dict(status='STARTING', device='cuda', source_sha256=sha(__file__),
                   parent_sha256=expected, epochs_completed=0,
                   scientific_training_updates=0, discarded_preflight_updates=0,
                   heldout_access=False, elapsed_seconds=None)
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        try:
            assert torch.cuda.is_available()
            data = Sequences(ROLES[0], 1)
            index = next(i for i, row in enumerate(data.rows) if int(row['count']) == 3)
            items = [batch(item, 'cuda') for item in data[index]]
            net, saved = load(checkpoint, 'cuda', 'dual_context', 'all')
            assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 36554507
            parent = build('unet_mean').cuda().eval()
            parent.load_state_dict(saved['model'])
            net.eval()
            with torch.no_grad():
                outputs, counts = predict_bundle(net, items)
                max_delta = 0.
                for item, output in zip(items, outputs):
                    reference, parent_counts = predict(parent, item)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    torch.testing.assert_close(counts, parent_counts, rtol=0, atol=0)
                    max_delta = max(max_delta, float((output-reference).abs().max()))
            del parent, saved, outputs, reference, counts, parent_counts
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            net.train()
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5,
                                          weight_decay=1e-4, foreach=False)
            chosen = {'encoder': net.down[0][0].weight,
                      'decoder': net.up[-1][0].weight,
                      'output': net.output.weight,
                      'cross_window': net.gates.expand.weight,
                      'ordered_power': net.ordered_projection.weight}
            before = {k: p.detach().clone() for k, p in chosen.items()}
            optimizer.zero_grad(set_to_none=True)
            values, _ = step_bundle(net, items, 1)
            gradients = {k: float(p.grad.norm()) if p.grad is not None else 0.
                         for k, p in chosen.items()}
            assert all(value > 0 for value in gradients.values()), gradients
            norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            changes = {k: float((p.detach()-before[k]).abs().max()) for k, p in chosen.items()}
            assert all(value > 0 for value in changes.values()), changes
            torch.cuda.synchronize()
            receipt.update(status='PASS', gpu=torch.cuda.get_device_name(),
                           trainable_parameters=36554507, sequence_index=index,
                           input_samples=191616, input_ms=1.91616,
                           initial_parent_max_difference=max_delta,
                           gradient_norms=gradients, max_parameter_changes=changes,
                           total_gradient_norm=norm,
                           preflight_loss_values=values,
                           discarded_preflight_updates=1,
                           max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                           max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                           weights_saved=False, weights_discarded=True,
                           interpretation='Feasibility only; no improvement or convergence claim')
        except Exception:
            receipt.update(status='FAILED', error=traceback.format_exc(), weights_saved=False)
            raise
        finally:
            receipt['elapsed_seconds'] = time.monotonic()-started
            OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
