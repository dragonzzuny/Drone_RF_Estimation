"""CPU saved-epoch validation, checkpoint, and optimizer audit; no new inference."""
import argparse
import numpy as np
import torch
from train import *


def main(epoch):
    torch.set_num_threads(1)
    event=read(OUT/f'EPOCH_{epoch:03d}.json')
    p=read(OUT/'PROTOCOL.json');verify(p)
    assert event['protocol_sha256']==sha256(OUT/'PROTOCOL.json')
    assert sha256(OUT/f'VALIDATION_{epoch:03d}.json')==event['validation_sha256']
    assert sha256(OUT/f'ACTUAL_{epoch:03d}.pt')==event['checkpoint_sha256']
    v=read(OUT/f'VALIDATION_{epoch:03d}.json');dev=Mixtures(ROLES[1])
    train=Mixtures(ROLES[0],epoch,features=False)
    assert train.rows_hash==event['training_rows_sha256']
    assert v['rows_sha256']==dev.rows_hash and len(v['rows'])==630
    assert aggregate(v['rows'])=={k:v[k] for k in ('by_count','selection_nmse')}
    for i,r in enumerate(v['rows']):
        row=dev.rows[i];count=int(row['count'])
        assert r['index']==i and r['count']==count
        assert r['clip_ids']==[dev.library.clips[int(j)]['clip_id'] for j in row['indices'][:count]]
        assert r['nominal_levels_db']==row['levels'][:count].tolist()
        assert len(r['nmse'])==count and r['weakest_index']==int(np.argmin(r['reference_power']))
        assert sorted(r['assignment'])==[0,1,2]
    saved=torch.load(OUT/'LAST.pt',map_location='cpu',weights_only=False,mmap=True)
    actual=torch.load(OUT/f'ACTUAL_{epoch:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==epoch and saved['updates']==epoch*75 and saved['epoch_evaluated']
    assert saved['protocol_sha256']==event['protocol_sha256']==actual['protocol_sha256']
    assert {int(x['step']) for x in saved['optimizer']['state'].values()}=={epoch*75}
    assert len(event['gradient_norms'])==75 and np.isfinite(event['gradient_norms']).all()
    for slot in saved['optimizer']['state'].values():
        assert all(torch.isfinite(x).all() for x in slot.values() if torch.is_tensor(x))
    for name,tensor in actual['model'].items():
        assert torch.isfinite(tensor).all() and torch.equal(tensor,saved['model'][name])
    assert sum(t.numel() for t in actual['model'].values())==4601867
    best=torch.load(OUT/'BEST.pt',map_location='cpu',weights_only=False,mmap=True)
    be=event['best']['epoch']
    selected=torch.load(OUT/(f'ACTUAL_{be:03d}.pt' if be else 'INITIAL.pt'),
                        map_location='cpu',weights_only=False,mmap=True)
    assert best['protocol_sha256']==event['protocol_sha256'] and best['epoch']==be
    assert all(torch.equal(t,selected['model'][k]) for k,t in best['model'].items())
    write_json(OUT/f'AUDIT_{epoch:03d}.json',dict(status='PASS',epoch=epoch,
        event_sha256=sha256(OUT/f'EPOCH_{epoch:03d}.json'),protocol_sha256=event['protocol_sha256']))
    write_json(PUBLIC/f'WAVENET100_E{epoch:03d}.json',event)
    lines=['# 동일10MHz 시간 분리: 완료 epoch','',
        'B 진단: 같은 원녹음의 겹치지 않는 시간 구간. 독립 녹음 평가가 아니다. 새 초기화·총100epoch,8위상 추론.','',
        '| Epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 최약 NMSE 2/3 ↓ | 선택 epoch |','|---:|---|---|---|---:|']
    for k in range(1,epoch+1):
        assert read(OUT/f'AUDIT_{k:03d}.json')['status']=='PASS'
        e=read(OUT/f'EPOCH_{k:03d}.json');a,b=e['validation']['by_count'][1:]
        si=lambda g:'미정의' if g['mean_si_sdr'] is None else f"{g['mean_si_sdr']:.3f}"
        lines.append(f"|{k}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{si(a)}/{si(b)}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{e['best']['epoch']}|")
    (PUBLIC/'WAVENET100_EPOCH_REPORTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(f'CPU_AUDIT_PASS epoch={epoch}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,required=True)
    main(parser.parse_args().epoch)
