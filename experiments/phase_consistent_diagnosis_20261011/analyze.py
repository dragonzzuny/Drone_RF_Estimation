"""Read-only posthoc diagnosis of sealed e0/e6 rows; no I/Q or torch access."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'local/phase_consistent_20261011_v1'
PUBLIC = ROOT / 'reports/2026-10-11'
HASHES = {}


def read(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def group_summary(key, pairs):
    a = [statistics.mean(x['nmse']) for x, _ in pairs]
    b = [statistics.mean(y['nmse']) for _, y in pairs]
    return dict(group=key, cases=len(pairs), initial_nmse=statistics.mean(a),
                epoch6_nmse=statistics.mean(b), delta=statistics.mean(v-u for u,v in zip(a,b)),
                worsened_cases=sum(v>u for u,v in zip(a,b)))


def main():
    protocol = read(RUN / 'PROTOCOL.json')
    initial = read(RUN / 'EIGHT_000.json')
    latest = read(RUN / 'EIGHT_006.json')
    assert len(initial['rows']) == len(latest['rows']) == 630
    curves = []
    for epoch in range(1, 7):
        audit = read(PUBLIC / f'PHASE_CONSISTENT_AUDIT_E{epoch:03d}.json')
        assert audit['status'] == 'PASS' and audit['epoch'] == epoch
        assert audit['protocol_sha256'] == HASHES[str((RUN/'PROTOCOL.json').relative_to(ROOT))]
        t = read(RUN / f'TRAIN_{epoch:03d}.json')
        d = read(RUN / f'DIAGNOSTIC_{epoch:03d}.json')
        e = read(RUN / f'EPOCH_{epoch:03d}.json')
        assert audit['checkpoint_sha256'] == e['checkpoint_sha256']
        assert t['schedule'] == protocol['train_schedules'][epoch-1]
        assert t['updates'] == epoch*75 and t['examples'] == 2400
        norms=t['gradient_norms']
        curves.append(dict(epoch=epoch, schedule=t['schedule'], online_train_loss=t['mean_loss'],
                           dev_diagnostic=d, dev_by_count=e['eight']['by_count'],
                           gradient_norm_median=statistics.median(norms),
                           clipped_fraction=sum(g>1 for g in norms)/len(norms)))
    compositions=collections.defaultdict(list)
    spreads=collections.defaultdict(list)
    counts=collections.defaultdict(list)
    for x,y in zip(initial['rows'],latest['rows']):
        for k in ('index','count','categories','pack_ids','reference_power','nominal_levels_db'):
            assert x[k] == y[k], (k,x['index'])
        assert len(x['nmse']) == len(y['nmse']) == x['count']
        assert all(math.isfinite(v) for v in x['nmse']+y['nmse'])
        counts[x['count']].append((x,y))
        compositions[(x['count'],tuple(x['categories']))].append((x,y))
        if x['count']>1:
            spread=10*math.log10(max(x['reference_power'])/min(x['reference_power']))
            label='0-10dB' if spread<10 else '10-20dB' if spread<20 else '20-30dB' if spread<30 else '30+dB'
            spreads[(x['count'],label)].append((x,y))
    summary=[]
    for k,pairs in sorted(counts.items()):
        a=initial['by_count'][k-1];b=latest['by_count'][k-1]
        row=group_summary(k,pairs)
        assert row['cases']==210
        assert math.isclose(row['initial_nmse'],a['mean_nmse'],abs_tol=1e-12)
        assert math.isclose(row['epoch6_nmse'],b['mean_nmse'],abs_tol=1e-12)
        row.update(relative_change_percent=100*row['delta']/row['initial_nmse'],
                   initial_weakest_nmse=a['weakest_nmse'],epoch6_weakest_nmse=b['weakest_nmse'])
        summary.append(row)
    t1=curves[0]['online_train_loss'];t6=curves[5]['online_train_loss']
    assert curves[0]['schedule']==curves[5]['schedule']==1
    changes={k:t6[k]-t1[k] for k in t1}
    assert math.isclose(changes['total'],changes['waveform']+.1*changes['consistency']+.1*changes['count'],abs_tol=1e-8)
    prior=read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_FINAL.json')
    old_control=prior['histories']['retained_unet'][:2]
    data=dict(status='POSTHOC_SAVED_ROW_AUDIT_PASS',evaluated_epoch=6,
              source_sha256=HASHES,by_count=summary,
              by_composition=[group_summary([k[0],list(k[1])],v) for k,v in sorted(compositions.items())],
              by_actual_reference_power_spread=[group_summary(list(k),v) for k,v in sorted(spreads.items())],
              learning_curves=curves,same_schedule_online_loss_change_1_to_6=changes,
              historical_single_view_control=old_control,
              heldout_read=False,raw_iq_read=False,new_inference=False,new_updates=0,
              caveats=['Online training losses use changing weights, not fixed-checkpoint evaluation.',
                       'Same-schedule e1/e6 receipt comparison is descriptive, not a causal ablation.',
                       '630 windows derive from five DEV recording groups, not 630 independent recordings.',
                       'Scalar loss magnitudes do not establish gradient magnitudes or interference.',
                       'Historical control uses single inference and a different learning rate/budget.',
                       'Power-spread bins are posthoc and not received SNR; no condition was excluded.'])
    out=PUBLIC/'PHASE_CONSISTENT_E006_DIAGNOSIS.json'
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=data['status'],by_count=summary,
                         same_schedule_online_loss_change=changes),ensure_ascii=False))


if __name__=='__main__':
    main()
