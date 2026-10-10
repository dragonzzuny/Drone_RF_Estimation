"""Metadata-only scope audit; no I/Q or validation examples are opened."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=json.loads((ROOT/'local/tfgridnet_continuation_20261011_v1/PROTOCOL.json').read_text())
    directory=Path(p['preparation']);prep=directory/'PREPARATION.json';manifest=directory/'NATIVE_MANIFEST.json'
    config=json.loads(prep.read_text());clips=json.loads(manifest.read_text())['clips']
    schedule=Path(config['original_preparation'])/'TRAIN.npy';rows=np.load(schedule,allow_pickle=False)
    rows=rows[rows['epoch']==1];fit_indices=set(p['train_indices'])
    assert sha(manifest)==config['manifest_sha256'] and sha(schedule)==config['schedule_sha256']['TRAIN.npy']
    def components(index):
        row=rows[index];start=min(max(int(row['crop_start']),4096),2**21-4096-63872)-4096
        result=[]
        for k in row['indices'][:int(row['count'])]:
            c=clips[int(k)];assert c['role']=='train_pack' and c['fs_hz']==100000000
            result.append(dict(source_path=c['source_path'],pack_id=c['pack_id'],category=c['category'],
                center_hz=c['common_center_hz'],start=int(c['offset_samples'])+start,end=int(c['offset_samples'])+start+63872))
        return result
    fitted=[c for i in sorted(fit_indices) for c in components(i)]
    categories={c['category'] for c in fitted};packs={c['pack_id'] for c in fitted};centers={c['center_hz'] for c in fitted}
    output=[]
    for index in range(48):
        if int(rows[index]['count']) not in (2,3):continue
        cs=components(index)
        for c in cs:
            spans=sorted((max(c['start'],f['start']),min(c['end'],f['end'])) for f in fitted
                if c['source_path']==f['source_path'] and max(c['start'],f['start'])<min(c['end'],f['end']))
            merged=[]
            for a,b in spans:
                if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
                else:merged.append([a,b])
            c['fraction_overlap_with_fit_waveform_windows']=sum(b-a for a,b in merged)/63872
        output.append(dict(index=index,count=int(rows[index]['count']),subset='fit4' if index in fit_indices else 'outside_fit4',
            all_categories_in_fit=all(c['category'] in categories for c in cs),
            all_record_groups_in_fit=all(c['pack_id'] in packs for c in cs),
            all_bands_in_fit=all(c['center_hz'] in centers for c in cs),
            any_waveform_sample_overlap=any(c['fraction_overlap_with_fit_waveform_windows']>0 for c in cs),components=cs))
    outside=[r for r in output if r['subset']=='outside_fit4']
    result=dict(status='COMPLETE',rows=output,summary=dict(fit_categories=sorted(categories),fit_record_groups=sorted(packs),
        fit_common_centers_hz=sorted(centers),outside_mixtures=len(outside),
        outside_with_unfitted_category=sum(not r['all_categories_in_fit'] for r in outside),
        outside_with_unfitted_record_group=sum(not r['all_record_groups_in_fit'] for r in outside),
        outside_with_waveform_sample_overlap=sum(r['any_waveform_sample_overlap'] for r in outside)),
        files={str(x):sha(x) for x in [Path(__file__),prep,manifest,schedule]},
        iq_read=False,validation_examples_read=False,heldout_read=False,
        scope='raw sample interval overlap from metadata; RF filtering may add dependence beyond nominal intervals; long context overlap not excluded',time=time.time())
    target=ROOT/'reports/2026-10-11/TFGRIDNET_OUTSIDE_FIT_SCOPE.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
