"""Full frozen development validation, whole-window PIT and explicit metrics."""
import time
import numpy as np
import torch

from late_models import predict, LEGACY
import sys
sys.path.insert(0,str(LEGACY))
from study import finite_values, value_status
from long_data import batch
from drone_rf.waveform import waveform_metrics
from drone_rf.context_training_data import write_json


@torch.no_grad()
def evaluate(net, dataset, path, epoch):
    net.eval()
    rows=[]
    for index in range(len(dataset)):
        item=batch(dataset[index],'cuda')
        estimate,logits=predict(net,item)
        m=waveform_metrics(estimate,item['references'],item['active'],item['mixture'])
        active=item['active'][0]; count=int(item['construction_count'][0])
        source_row=dataset.rows[index]
        clips=[dataset.library.clips[int(i)] for i in source_row['indices'][:count]]
        si=m['si_sdr'][0][active]; gain=si-m['input_si_sdr'][0][active]
        power=m['reference_power'][0][active]
        row=dict(index=index,count=count,categories=[c['category'] for c in clips],
            pack_ids=[c['pack_id'] for c in clips],nominal_levels_db=source_row['levels'][:count].tolist(),
            reference_power=power.cpu().tolist(),nmse=finite_values(m['nmse'][0][active]),
            assignment=m['assignment'][0].cpu().tolist(),si_sdr=finite_values(si),
            si_status=value_status(si),input_si_sdr=finite_values(m['input_si_sdr'][0][active]),
            si_sdr_gain=finite_values(gain),weakest_index=int(power.argmin()),
            predicted_count=int(logits.argmax(-1)[0])+1,
            inactive_leak=float(m['inactive_leak'][0].sum()),background_nmse=float(m['background_nmse'][0]),
            sum_relative_error=float(m['sum_relative_error'][0]))
        if any(x is None for x in row['nmse']) or row['sum_relative_error']>1e-9:
            raise RuntimeError('Invalid waveform metric or mixture sum')
        rows.append(row)
        if index%50==0:
            write_json(path.parent.parent/'PROGRESS.json',dict(stage='VALIDATION',arm=path.parent.name,
                epoch=epoch,examples=index+1,total=len(dataset),time=time.time()))
    groups=[]
    for count in (1,2,3):
        group=[r for r in rows if r['count']==count]
        if len(group)!=210:
            raise RuntimeError('Validation count stratum changed')
        def average(key):
            values=[v for r in group for v in r[key]]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        groups.append(dict(count=count,cases=len(group),mean_nmse=average('nmse'),
            mean_si_sdr=average('si_sdr'),mean_si_sdr_gain=average('si_sdr_gain') if count>1 else None,
            nonfinite_si_sdr=sum(v is None for r in group for v in r['si_sdr']),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            construction_count_accuracy=float(np.mean([r['predicted_count']==count for r in group]))))
    result=dict(epoch=epoch,by_count=groups,rows=rows,
        selection_nmse=float(np.mean([g['mean_nmse'] for g in groups if g['count']>1])),
        independent_test=False,physical_aircraft_count=False,whole_record_tracking=False)
    write_json(path,result)
    return result
