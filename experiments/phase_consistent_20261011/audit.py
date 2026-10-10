"""CPU receipt/tensor/row audit. This is not independent waveform reinference."""
import argparse
import math
import statistics
import torch
from common import OUT,PUBLIC,ROOT,w

def audit(epoch):
    torch.set_num_threads(2)
    p=w.read(OUT/'PROTOCOL.json');ph=w.digest(OUT/'PROTOCOL.json')
    event=w.read(OUT/f'EPOCH_{epoch:03d}.json')
    assert event['protocol_sha256']==ph and event['updates']==75*epoch
    for rel,h in p['source_sha256'].items():assert w.digest(ROOT/rel)==h
    cp=OUT/f'ACTUAL_{epoch:03d}.pt';assert w.digest(cp)==event['checkpoint_sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['epoch']==epoch and saved['updates']==epoch*75 and saved['protocol_sha256']==ph
    assert sum(v.numel() for v in saved['model'].values())==32142859
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={75*epoch}
    assert all(g['lr']==p['learning_rate'] and g['weight_decay']==p['weight_decay'] for g in saved['optimizer']['param_groups'])
    del saved
    initial=w.read(OUT/'EIGHT_000.json');fields=('index','count','categories','pack_ids','nominal_levels_db','reference_power')
    for kind in ('SINGLE','EIGHT'):
        report=w.read(OUT/f'{kind}_{epoch:03d}.json'); rows=report['rows']
        assert len(rows)==630 and [r['index'] for r in rows]==list(range(630))
        assert report['by_count']==event[kind.lower()]['by_count']
        for a,b in zip(rows,initial['rows']):
            assert all(a[k]==b[k] for k in fields)
            assert a['sum_relative_error']<1e-9
        for k in (1,2,3):
            r=[v for v in rows if v['count']==k];assert len(r)==210
            g=report['by_count'][k-1]
            nm=[v for x in r for v in x['nmse']];si=[v for x in r for v in x['si_sdr']]
            weak=[x['nmse'][x['weakest_index']] for x in r]
            assert math.isclose(statistics.mean(nm),g['mean_nmse'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(statistics.mean(weak),g['weakest_nmse'],rel_tol=1e-12,abs_tol=1e-12)
            if all(v is not None for v in si):
                assert math.isclose(statistics.mean(si),g['mean_si_sdr'],rel_tol=1e-12,abs_tol=1e-12)
    baseline=w.read(__import__('pathlib').Path(p['baseline']))['eight_phase']
    conditions=[]
    for a,b in zip(event['eight']['by_count'][1:],baseline['by_count'][1:]):
        conditions.extend([a['mean_nmse']<b['mean_nmse'],a['mean_si_sdr'] is not None and a['mean_si_sdr']>b['mean_si_sdr'],
                           a['weakest_nmse']<=b['weakest_nmse']])
    assert event['candidate']==all(conditions)
    receipt=dict(status='PASS',epoch=epoch,updates=75*epoch,checkpoint_sha256=event['checkpoint_sha256'],
        protocol_sha256=ph,independent_cpu_waveform_reinference=False,
        checked='source hashes, all checkpoint tensors, optimizer steps, all1260rows, independent aggregates, candidate criterion')
    w.write(PUBLIC/f'PHASE_CONSISTENT_AUDIT_E{epoch:03d}.json',receipt)
    lines=[f'# 8위상 일관성 U-Net 추가epoch{epoch}','',
           f'학습75업데이트 완료, 누적{75*epoch}회. CPU 저장점·행/집계 감사PASS. 파형의 독립 CPU 재추론은 아님.','',
           '|추론|성분수|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|','|---|---:|---:|---:|---:|']
    for kind in ('single','eight'):
        for r in event[kind]['by_count']:
            score='비유한값 있음' if r['mean_si_sdr'] is None else f"{r['mean_si_sdr']:.6f}"
            lines.append(f"|{kind}|{r['count']}|{r['mean_nmse']:.8f}|{score}|{r['weakest_nmse']:.8f}|")
    lines += ['',f"고정8위상 부모 대비 공동 기준: {event['candidate']}. 연구 내부 선택epoch: {event['best']['epoch']}.",
              f"위상일관성 진단 평균: 초기{event['baseline_diagnostic']['mean_consistency']:.8f} → {event['diagnostic']['mean_consistency']:.8f}.",
              '한 seed의 반복DEV이며 미개봉 확인자료는 사용하지 않았다. 현재 최선 manifest를 자동 변경하지 않는다.','']
    w.write(PUBLIC/f'PHASE_CONSISTENT_E{epoch:03d}_KO.md','\n'.join(lines))
    print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',required=True,type=int);a=p.parse_args();audit(a.epoch)
