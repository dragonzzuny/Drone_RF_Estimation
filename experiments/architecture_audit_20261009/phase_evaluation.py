"""Same complex waveform evaluation on the fixed 630 native mixtures."""
import os
from pathlib import Path
import statistics
import time

import torch

import phase_data as data
from native_data import write_json
from drone_rf.waveform import waveform_metrics
from study import finite_values, value_status
import watch_epochs


def row_metrics(predicted, item, source_row, clips, index):
    estimates, logits = predicted['estimates'], predicted['count_logits']
    score = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
    active, count = item['active'][0], int(item['construction_count'][0])
    power, si = score['reference_power'][0][active], score['si_sdr'][0][active]
    gain = si - score['input_si_sdr'][0][active]
    return dict(index=index, count=count, categories=[c['category'] for c in clips],
        pack_ids=[c['pack_id'] for c in clips], nominal_levels_db=source_row['levels'][:count].tolist(),
        reference_power=power.cpu().tolist(), assignment=score['assignment'][0].cpu().tolist(),
        nmse=finite_values(score['nmse'][0][active]), si_sdr=finite_values(si), si_status=value_status(si),
        input_si_sdr=finite_values(score['input_si_sdr'][0][active]),
        si_sdr_gain=finite_values(gain), gain_status=value_status(gain), weakest_index=int(power.argmin()),
        all_sources_gain_positive=bool(torch.all(torch.isfinite(gain) & (gain > 0))) if count > 1 else None,
        predicted_count=int(logits.argmax(-1)[0])+1, inactive_leak=float(score['inactive_leak'][0].sum()),
        background_nmse=float(score['background_nmse'][0]),
        sum_relative_error=float(score['sum_relative_error'][0]))


def aggregate(rows, epoch):
    groups = []
    for count in (1, 2, 3):
        subset = [r for r in rows if r['count'] == count]
        if len(subset) != 210:
            raise ValueError('Validation count strata changed')
        def average(key):
            values = [v for r in subset for v in r[key]]
            return statistics.mean(values) if all(v is not None for v in values) else None
        groups.append(dict(count=count, cases=210, mean_nmse=average('nmse'),
            mean_si_sdr=average('si_sdr'), mean_si_sdr_gain=average('si_sdr_gain') if count > 1 else None,
            nonfinite_si_sdr=sum(v is None for r in subset for v in r['si_sdr']),
            nonfinite_si_sdr_gain=sum(v is None for r in subset for v in r['si_sdr_gain']),
            weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in subset),
            construction_count_accuracy=statistics.mean(r['predicted_count']==count for r in subset),
            all_sources_gain_positive=statistics.mean(r['all_sources_gain_positive'] for r in subset) if count > 1 else None))
    return dict(epoch=epoch, by_count=groups, rows=rows,
        selection_nmse=statistics.mean(g['mean_nmse'] for g in groups if g['count'] > 1),
        independent_test=False, physical_aircraft_count=False, whole_record_tracking=False,
        native_center_offsets_preserved=True,
        reconstruction_target='common-RF-band-limited recorded contributions, not complete raw records')


@torch.no_grad()
def evaluate(net, dataset, path, epoch, identities):
    net.eval()
    rows, began = [], time.time()
    root = path.parent.parent
    for index in range(len(dataset)):
        item = data.batch(data.example(dataset, index), 'cuda')
        source = dataset.rows[index]
        clips = [dataset.library.clips[int(i)] for i in source['indices'][:int(source['count'])]]
        rows.append(row_metrics(data.predict(net, item), item, source, clips, index))
        if index % 50 == 0:
            write_json(root/'STATE.json', dict(status='VALIDATING', arm=net.scope, epoch=epoch,
                examples=index+1, total=len(dataset), seconds=time.time()-began,
                pid=os.getpid(), time=time.time()))
    result = aggregate(rows, epoch)
    write_json(path, result)
    # Verify all identities, finite NMSE, mixture sum and recomputed aggregates
    # against the existing main study without opening a new evaluation split.
    checked, _ = watch_epochs.validate(path, identities)
    return result
