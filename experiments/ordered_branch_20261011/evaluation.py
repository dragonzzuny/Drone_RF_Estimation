"""Paired one/four-phase DEV evaluation sharing the zero-phase pass."""
import copy
import importlib.util
import numpy as np
import torch
import architecture as model
from drone_rf.waveform import waveform_metrics
from study import finite_values,value_status

ROOT,w,worker=model.ROOT,model.w,model.worker
PHASE_SOURCE=ROOT/'experiments/rfuav_phase_average_20261009/phase_core.py'
spec=importlib.util.spec_from_file_location('ordered_branch_phase',PHASE_SOURCE)
phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)


def summarize(rows,epoch=1):
    assert len(rows)==630 and all(r['sum_relative_error']<1e-9 for r in rows)
    groups=[]
    for k in (1,2,3):
        selected=[r for r in rows if r['count']==k];assert len(selected)==210
        nm=[v for r in selected for v in r['nmse']];si=[v for r in selected for v in r['si_sdr']]
        assert all(v is not None and np.isfinite(v) for v in nm+si)
        groups.append(dict(count=k,cases=210,mean_nmse=float(np.mean(nm)),median_nmse=float(np.median(nm)),
            mean_si_sdr=float(np.mean(si)),nonfinite_si_sdr=0,
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in selected])),
            construction_count_accuracy=float(np.mean([r['count']==r['predicted_count'] for r in selected]))))
    return dict(epoch=epoch,by_count=groups,rows=rows,selection_nmse=float(np.mean([g['mean_nmse'] for g in groups[1:]])),
        independent_test=False,physical_aircraft_count=False,whole_record_tracking=False)


def compare(candidate,parent):
    rows=[]
    for a,b in zip(parent['by_count'][1:],candidate['by_count'][1:]):
        assert a['count']==b['count']
        rows.append(dict(count=a['count'],nmse=b['mean_nmse']<a['mean_nmse'],
            si=b['mean_si_sdr']>a['mean_si_sdr'],weak=b['weakest_nmse']<=a['weakest_nmse']))
    return rows


@torch.no_grad()
def evaluate(net,data,baseline,state,out):
    net.eval();single=[];averaged=[]
    for i in range(len(data)):
        if i%25==0:state('VALIDATING',epoch=1,case=i,total=630,updates=75)
        item=worker.fit.base.batch([data[i]])
        first,mean,logits,diag=phase.phase_predictions(net,worker.predict,item)
        for predicted,destination in ((first,single),(mean,averaged)):
            m=waveform_metrics(predicted,item['references'],item['active'],item['mixture'])
            active=item['active'][0];power=m['reference_power'][0][active]
            row=copy.deepcopy(baseline['rows'][i])
            assert row['index']==i and row['count']==int(item['construction_count'][0])
            assert row['reference_power']==power.tolist()
            si=m['si_sdr'][0][active];gain=si-m['input_si_sdr'][0][active]
            row.update(assignment=m['assignment'][0].tolist(),nmse=finite_values(m['nmse'][0][active]),
                si_sdr=finite_values(si),si_status=value_status(si),si_sdr_gain=finite_values(gain),
                gain_status=value_status(gain),weakest_index=int(power.argmin()),
                predicted_count=int(logits.argmax(-1)[0])+1,inactive_leak=float(m['inactive_leak'][0].sum()),
                background_nmse=float(m['background_nmse'][0]),sum_relative_error=float(m['sum_relative_error'][0]),
                phase_details=diag if destination is averaged else dict(forward_passes=1))
            assert row['predicted_count']==baseline['rows'][i]['predicted_count']
            destination.append(row)
    a,b=summarize(single),summarize(averaged)
    w.write(out/'SINGLE_001.json',a);w.write(out/'FOUR_PHASE_001.json',b)
    return a,b
