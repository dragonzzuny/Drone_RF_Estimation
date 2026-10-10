"""Original native DEV scoring schema with an explicit prediction function.

Keeps the frozen dataset and metric implementation; does not monkey-patch the
active experiment's modules. All per-source rows, including failures, retained.
"""
import numpy as np
import torch
from study import finite_values,value_status
from drone_rf.waveform import waveform_metrics


@torch.no_grad()
def validate(net,data,path,epoch,predict,worker):
    net.eval();rows=[]
    for index in range(len(data)):
        item=worker.fit.base.batch([data[index]])
        estimates,logits=predict(net,item)
        m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
        active=item['active'][0];count=int(item['construction_count'][0])
        power=m['reference_power'][0][active];si=m['si_sdr'][0][active]
        gain=si-m['input_si_sdr'][0][active];source=data.rows[index]
        clips=[data.library.clips[int(i)] for i in source['indices'][:count]]
        rows.append(dict(index=index,count=count,categories=[c['category'] for c in clips],
            pack_ids=[c['pack_id'] for c in clips],nominal_levels_db=source['levels'][:count].tolist(),
            reference_power=power.cpu().tolist(),assignment=m['assignment'][0].cpu().tolist(),
            nmse=finite_values(m['nmse'][0][active]),si_sdr=finite_values(si),si_status=value_status(si),
            input_si_sdr=finite_values(m['input_si_sdr'][0][active]),si_sdr_gain=finite_values(gain),gain_status=value_status(gain),
            weakest_index=int(power.argmin()),
            all_sources_gain_positive=bool(torch.all(torch.isfinite(gain)&(gain>0))) if count>1 else None,
            predicted_count=int(logits.argmax(-1)[0])+1,inactive_leak=float(m['inactive_leak'][0].sum()),
            background_nmse=float(m['background_nmse'][0]),sum_relative_error=float(m['sum_relative_error'][0])))
    assert len(rows)==630 and all(r['sum_relative_error']<1e-9 for r in rows)
    groups=[]
    for count in (1,2,3):
        selected=[r for r in rows if r['count']==count];assert len(selected)==210
        def mean(key):
            v=[x for r in selected for x in r[key]]
            return float(np.mean(v)) if all(x is not None for x in v) else None
        groups.append(dict(count=count,cases=210,mean_nmse=mean('nmse'),mean_si_sdr=mean('si_sdr'),
            nonfinite_si_sdr=sum(x is None for r in selected for x in r['si_sdr']),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in selected])),
            construction_count_accuracy=float(np.mean([r['count']==r['predicted_count'] for r in selected]))))
    result=dict(epoch=epoch,by_count=groups,selection_nmse=float(np.mean([g['mean_nmse'] for g in groups[1:]])),
        native_center_offsets_preserved=True,reconstruction_target='common-RF-band-limited recorded contributions, not complete raw records',
        independent_test=False,physical_aircraft_count=False,whole_record_tracking=False,rows=rows)
    worker.watch.write(path,result);return result
