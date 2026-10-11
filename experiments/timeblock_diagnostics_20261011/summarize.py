"""Descriptive saved-row audit; no new inference, selection or training changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'local/bandwidth_timeblock_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def describe(values):
    a=np.asarray(values,dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def main(epoch):
    initial_path=RUN/'VALIDATION_000.json';current_path=RUN/f'VALIDATION_{epoch:03d}.json'
    initial=json.loads(initial_path.read_text());current=json.loads(current_path.read_text())
    assert initial['rows_sha256']==current['rows_sha256']
    if epoch:
        assert json.loads((RUN/f'AUDIT_{epoch:03d}.json').read_text())['status']=='PASS'
    records=[]
    for a,b in zip(initial['rows'],current['rows']):
        assert all(a[k]==b[k] for k in ('index','count','clip_ids','reference_power','weakest_index'))
        if b['count']==1:continue
        spread=10*np.log10(max(b['reference_power'])/min(b['reference_power']))
        band='<=10dB' if spread<=10 else '10-20dB' if spread<=20 else '>20dB'
        records.append(dict(before=a,after=b,spread_db=float(spread),spread_group=band))
    groups=[]
    for count in (2,3):
        rows=[r for r in records if r['after']['count']==count]
        summaries=[('all',rows)]
        summaries += [(name,[r for r in rows if r['spread_group']==name]) for name in ('<=10dB','10-20dB','>20dB')]
        compositions=sorted({tuple(r['after']['categories']) for r in rows})
        summaries += [(' + '.join(c),[r for r in rows if tuple(r['after']['categories'])==c]) for c in compositions]
        for name, selected in summaries:
            if not selected:continue
            output=dict(count=count,stratum=name,cases=len(selected),
                local_strong_weak_power_db=describe([r['spread_db'] for r in selected]))
            for key in ('before','after'):
                output[key]=dict(nmse=describe([v for r in selected for v in r[key]['nmse']]),
                    weakest_nmse=describe([r[key]['nmse'][r[key]['weakest_index']] for r in selected]))
            output['sources_nmse_improved']=sum(v<w for r in selected for v,w in zip(r['after']['nmse'],r['before']['nmse']))
            groups.append(output)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='SAVED_ROWS_DESCRIPTIVE_AUDIT_PASS',epoch=epoch,groups=groups,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (initial_path,current_path)},source_sha256=sha(Path(__file__)),
        new_inference=False,training_or_selection_changed=False,
        independent_recording_validation=False,
        limitation='descriptive development-window statistics, not independent records or causal bandwidth attribution')
    path=PUBLIC/f'TIMEBLOCK_DISTRIBUTION_E{epoch:03d}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps([g for g in groups if g['stratum']=='all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True)
    main(p.parse_args().epoch)
