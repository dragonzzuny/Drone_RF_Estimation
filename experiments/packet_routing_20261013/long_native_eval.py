"""The already fixed 24 sequences, processed as whole contiguous waveforms.

This is an additional diagnostic, not a larger independent test set. The
original short630 and overlap-merged long24 remain separate comparisons.
"""
import torch
from checkpoint_blocks import combine_items
from sequence_data import Sequences, ROLES
from sequence_eval import score, summarize
from models import predict
from study import batch
from core import predict_eight


@torch.no_grad()
def evaluate(net, device, progress=None):
    net.eval()
    data = Sequences(ROLES[1])
    rows = []
    for i in range(len(data)):
        item = combine_items([batch(chunk, device) for chunk in data[i]])
        _, estimate, logits, _ = predict_eight(net, predict, item)
        count = int(item['construction_count'][0])
        row = score(estimate, item['references'], item['mixture'], count, logits)
        raw = data.rows[i]
        row.update(index=i, categories=[data.library.clips[int(j)]['category']
                                       for j in raw['indices'][:count]],
                   nominal_levels_db=raw['levels'][:count].tolist())
        rows.append(row)
        if progress:
            progress('NATIVE_LONG_DEV', case=i+1, total=len(data))
    return dict(rows=rows, **summarize(rows),
                waveform_samples=191616, duration_ms=1.91616,
                inference='Single contiguous input, unchanged C8, no targets in forward',
                caveat='Same fixed24 dependent development sequences, not new recordings')
