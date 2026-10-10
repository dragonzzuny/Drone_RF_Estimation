"""Fixed 630 DEV cases, single and eight views from identical weights."""
import copy
import numpy as np
import torch
from common import worker
from drone_rf.waveform import waveform_metrics
from study import finite_values, value_status
import objective as obj

def summarize(rows, epoch):
    assert len(rows) == 630
    groups = []
    for k in (1, 2, 3):
        selected = [r for r in rows if r['count'] == k]
        assert len(selected) == 210
        nm = [v for r in selected for v in r['nmse']]
        si = [v for r in selected for v in r['si_sdr']]
        assert all(v is not None and np.isfinite(v) for v in nm)
        bad = sum(v is None or not np.isfinite(v) for v in si)
        groups.append(dict(count=k, cases=210, mean_nmse=float(np.mean(nm)),
            median_nmse=float(np.median(nm)), mean_si_sdr=None if bad else float(np.mean(si)),
            nonfinite_si_sdr=bad,
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in selected])),
            construction_count_accuracy=float(np.mean([r['predicted_count']==k for r in selected]))))
    return dict(epoch=epoch, by_count=groups, rows=rows,
                selection_nmse=float(np.mean([g['mean_nmse'] for g in groups[1:]])),
                independent_test=False, physical_aircraft_count=False)

def acceptable(new, old):
    checks = []
    for a, b in zip(new['by_count'][1:], old['by_count'][1:]):
        checks.append(dict(count=a['count'], nmse=a['mean_nmse'] < b['mean_nmse'],
                           si=a['mean_si_sdr'] is not None and a['mean_si_sdr'] > b['mean_si_sdr'],
                           weak=a['weakest_nmse'] <= b['weakest_nmse']))
    return all(all(v for k,v in r.items() if k!='count') for r in checks), checks

@torch.no_grad()
def evaluate(net, data, baseline, epoch, state):
    net.eval(); single=[]; eight=[]; diagnostics=[]
    for i in range(len(data)):
        if i % 25 == 0:
            state('VALIDATING', epoch=epoch, updates=epoch*75, case=i, total=630)
        item=worker.fit.base.batch([data[i]])
        views, logits, orders=obj.collect(net,worker.predict,item)
        mean=obj.average(views)
        _, diagnostic=obj.losses(views,logits,item)
        diagnostics.append(diagnostic)
        for estimates,destination in ((views[0],single),(mean,eight)):
            m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
            active=item['active'][0]; power=m['reference_power'][0][active]
            row=copy.deepcopy(baseline['rows'][i])
            assert row['index']==i and row['count']==int(item['construction_count'][0])
            assert row['reference_power']==power.tolist()
            si=m['si_sdr'][0][active];gain=si-m['input_si_sdr'][0][active]
            row.update(assignment=m['assignment'][0].tolist(),nmse=finite_values(m['nmse'][0][active]),
                si_sdr=finite_values(si),si_status=value_status(si),si_sdr_gain=finite_values(gain),
                gain_status=value_status(gain),weakest_index=int(power.argmin()),
                predicted_count=int(logits.argmax(-1)[0])+1,inactive_leak=float(m['inactive_leak'][0].sum()),
                background_nmse=float(m['background_nmse'][0]),sum_relative_error=float(m['sum_relative_error'][0]),
                phase_details=dict(forward_passes=8 if destination is eight else 1,
                                   prediction_only_orders=orders if destination is eight else [[0,1,2]]))
            assert row['sum_relative_error']<1e-9
            destination.append(row)
    return summarize(single,epoch),summarize(eight,epoch),dict(
        mean_consistency=float(np.mean([r['consistency'] for r in diagnostics])),
        mean_waveform_objective=float(np.mean([r['waveform'] for r in diagnostics])),
        note='Reference-normalized diagnostic; targets used only after mixture-only forward')

