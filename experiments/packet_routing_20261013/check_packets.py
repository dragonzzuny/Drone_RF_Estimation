"""Apply the pre-existing fixed diagnostic receiver, without target-aided sync."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'
sys.path.insert(0,str(BASE/'receiver_pilot_v1'))
import receiver


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(d):
    while not (d/'VERIFICATION.json').exists():
        time.sleep(15)
    audit=json.loads((d/'VERIFICATION.json').read_text())
    assert audit['status']=='PASS' and audit['input_hashes_unchanged']
    assert sha(d/'ASSEMBLED.npz')==audit['artifacts']['ASSEMBLED.npz']
    frozen=json.loads((BASE/'receiver_pilot_v1/PLAN.json').read_text())
    for group in ('hashes','original_sources'):
        for name,h in frozen[group].items():assert sha(name)==h,name
    original=np.load(BASE/'retest_best_e079/SEPARATED_EVENT.npz')
    outputs=np.load(d/'ASSEMBLED.npz')['C8_hann']
    result=json.loads((d/'RESULT.json').read_text())
    refs=original['references'];mixture=original['mixture']
    tasks=[('reference_FPV',refs[0]),('reference_Mavic3Pro',refs[1]),('mixture',mixture)]
    tasks += [(f'output_{j}',outputs[j]) for j in range(3)]+[('background',outputs[3])]
    plan=dict(created_time=time.time(),epoch=result['epoch'],offset_hz=15000000,
        offset_role='fixed channel of the already exposed FPV event, not new blind frequency discovery',
        receiver_plan_sha256=sha(BASE/'receiver_pilot_v1/PLAN.json'),source_sha256=sha(__file__),
        assembly_sha256=sha(d/'ASSEMBLED.npz'),reference_event_sha256=sha(BASE/'retest_best_e079/SEPARATED_EVENT.npz'),
        no_reference_in_receiver=True,no_crc_guided_parameter_tuning=True,no_payload_fields_stored=True)
    assert not (d/'PACKET_PLAN.json').exists()
    write(d/'PACKET_PLAN.json',plan)
    a,b=round(.65e-3*receiver.T.A.RATE),round(1.35e-3*receiver.T.A.RATE)
    r=receiver.T.A.channelize(refs[0],receiver.T.FS,15000000)[a:b].astype(np.complex128)
    power=np.mean(abs(r)**2);rows=[]
    for label,x in tasks:
        began=time.time();decoded=receiver.receive(x,15000000)
        v=receiver.T.A.channelize(x,receiver.T.FS,15000000)[a:b].astype(np.complex128)
        alpha=np.vdot(r,v)/np.vdot(r,r)
        types=sorted({str(a['device_type']) for m in decoded.get('modes',[]) for a in m['attempts']
                      if a['crc16_ok'] and a['device_type'] is not None})
        rows.append(dict(label=label,status=decoded['status'],device_types=types,
            crc16=any(a['crc16_ok'] for m in decoded.get('modes',[]) for a in m['attempts']),
            crc24=any(a['crc24_ok'] for m in decoded.get('modes',[]) for a in m['attempts']),
            projection_coefficient_abs2=float(abs(alpha)**2),
            projected_FPV_power_ratio_db=float(10*np.log10(max(abs(alpha)**2,1e-30))),
            channel_power_ratio_to_FPV_db=float(10*np.log10(max(np.mean(abs(v)**2)/power,1e-30))),
            seconds=time.time()-began))
        write(d/'PACKET_RESULTS.json',dict(epoch=result['epoch'],rows=rows))
        print('PACKET',label,decoded['status'],types,round(rows[-1]['seconds'],2),flush=True)
    assert rows[0]['crc16'] and rows[2]['crc16'],'Existing positive inputs must remain decodable'
    write(d/'PACKET_AUDIT.json',dict(status='PASS',epoch=result['epoch'],inputs=len(rows),
        assignment=result['stages']['C8_hann']['metrics']['assignment'],
        results_sha256=sha(d/'PACKET_RESULTS.json'),plan_sha256=sha(d/'PACKET_PLAN.json'),
        note='CRC in mixture and duplicate outputs is not evidence of better detection or exclusive separation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    run(p.parse_args().directory)
