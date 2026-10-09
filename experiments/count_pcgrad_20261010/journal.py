"""Persist completed-epoch CPU audits/reports while GPU successors run.

This local journal does not send messages, commit files, change models or choose
new experiments. User-visible live chat updates are still supplied by the agent.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/2026-10-10'
STUDIES=[('count_pcgrad','count_pcgrad_20261010/audit.py','COUNT_PCGRAD'),
         ('count_cagrad','count_pcgrad_20261010/audit_cagrad.py','COUNT_CAGRAD'),
         ('wave_update_guard','count_pcgrad_20261010/audit_wave_guard.py','WAVE_UPDATE_GUARD'),
         ('tf_axis','tf_axis_20261010/audit.py','TF_AXIS')]


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run():
    journal=ROOT/'local/count_method_journal_20261010';journal.mkdir(exist_ok=True)
    processed=set();errors={}
    while True:
        states=[];changed=False
        for folder,audit,prefix in STUDIES:
            root=ROOT/f'local/{folder}_20261010_v1';state=root/'STATE.json'
            value=json.loads(state.read_text()) if state.exists() else dict(status='NOT_STARTED')
            states.append(dict(study=folder,**value))
            if (root/'COMPLETE.json').exists() and folder not in processed:
                output=PUBLIC/f'{prefix}_FINAL_AUDIT.json'
                if not output.exists():
                    with (journal/(folder+'_audit.log')).open('a') as log:
                        result=subprocess.run([sys.executable,str(ROOT/'experiments'/audit),'--run',str(root),'--output',str(output)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode!=0:
                        errors[folder]=dict(exit_code=result.returncode,log=str((journal/(folder+'_audit.log')).relative_to(ROOT)))
                if output.exists() and json.loads(output.read_text())['status']=='PASS':
                    print(dict(event='AUDITED_EPOCH',study=folder,epoch=1,time=time.time()),flush=True)
                processed.add(folder);changed=True
        if changed:
            with (journal/'report.log').open('a') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:errors['report']=dict(exit_code=result.returncode)
        done=all(s['status']=='COMPLETE' or s['status'].startswith('SKIPPED_') for s in states)
        write(journal/'STATE.json',dict(status='COMPLETE' if done else 'MONITORING',studies=states,errors=errors,pid=os.getpid(),time=time.time()))
        if done:return
        if any((ROOT/f"local/{s['study']}_20261010_v1/FAILURE.json").exists() for s in states):
            print(dict(event='WORKER_FAILURE_REQUIRES_REVIEW',states=states),flush=True)
            return
        time.sleep(10)


if __name__=='__main__':run()
