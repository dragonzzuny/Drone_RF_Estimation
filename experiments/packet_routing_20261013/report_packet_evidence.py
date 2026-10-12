"""Publish metrics and integrity receipts, excluding waveform/payload content."""
import argparse
import json
from pathlib import Path
from check_packets import sha,write,ROOT


def main(directory,name):
    read=lambda p:json.loads(p.read_text())
    verification=read(directory/'VERIFICATION.json')
    assert verification['status']=='PASS' and verification['input_hashes_unchanged']
    for file,h in verification['artifacts'].items():assert sha(directory/file)==h,file
    audit=read(directory/'PACKET_AUDIT.json')
    assert audit['status']=='PASS'
    assert sha(directory/'PACKET_RESULTS.json')==audit['results_sha256']
    assert sha(directory/'PACKET_PLAN.json')==audit['plan_sha256']
    result=read(directory/'RESULT.json');packet=read(directory/'PACKET_RESULTS.json')
    public=dict(status='VERIFIED',epoch=result['epoch'],
        stages={k:v.get('metrics',v) for k,v in result['stages'].items()},
        packet=packet,artifact_receipt=verification,packet_receipt=audit,
        evidence_scope='Same previously exposed FPV/Mavic3Pro2ms development event',
        no_reference_in_receiver=True,no_model_training_in_diagnosis=True,
        caveats=['Mixture already decodes FPV; no detector gain claim',
                 'Duplicate output CRC is not exclusive separation',
                 'Reference-based projection coefficients diagnose leakage; not used for prediction',
                 'Not independent recording validation'])
    write(ROOT/'reports/2026-10-13'/f'{name}.json',public)
    print('PACKET_EVIDENCE_VERIFIED',name,result['epoch'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();main(a.directory,a.name)
