"""Bounded local epoch/result reporting for the two authorized queued studies.

Writes auditable local reports and stdout events. It does not send messages,
claim chat notifications, modify models, or start additional experiments.
"""
import argparse
import json
import os
from pathlib import Path
import time
from report_rf_followups import phase,allocation


def save(prefix,result,markdown):
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    prefix.with_suffix('.md').write_text(markdown)


def main(phase_run,allocation_run):
    seen=set();last_phase=None;last_allocation=None
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        for root in (phase_run,allocation_run):
            if (root/'FAILURE.json').exists():
                print(json.dumps(dict(event='WORKER_FAILURE',run=str(root),failure=json.loads((root/'FAILURE.json').read_text()))),flush=True)
                return
        phase_files=sorted(phase_run.glob('*/COMPLETE.json'))
        signature=tuple((str(p),p.stat().st_mtime_ns) for p in phase_files)
        if signature and (signature!=last_phase or (phase_run/'COMPLETE.json').exists() and 'phase_final' not in seen):
            result,md=phase(phase_run);save(phase_run/'monitor/AUDITED_PROGRESS',result,md)
            last_phase=signature
            for model in result['models']:
                key=('phase',model['model'])
                if key not in seen:
                    print(json.dumps(dict(event='PHASE_MODEL_COMPLETE',model=model['model'],
                        criterion=model['directional_criterion'],change=model['change'])),flush=True);seen.add(key)
            if (phase_run/'COMPLETE.json').exists():
                seen.add('phase_final')
        for path in sorted(allocation_run.glob('*/EPOCH_*.json')):
            key=str(path)
            if key not in seen:
                record=json.loads(path.read_text());seen.add(key)
                print(json.dumps(dict(event='ALLOCATION_EPOCH_COMPLETE',arm=path.parent.name,
                    epoch=record['epoch'],best=record['best'],by_count=record['metrics']['by_count'])),flush=True)
        receipts=sorted(allocation_run.glob('*/EPOCH_*.json'))
        signature=tuple((str(p),p.stat().st_mtime_ns) for p in receipts)
        ready=all((allocation_run/a/'VALIDATION_000.json').exists() for a in ('waveform_only','waveform_allocation'))
        allocation_complete=(allocation_run/'COMPLETE.json').exists()
        if ready and signature and (signature!=last_allocation or allocation_complete and 'allocation_final' not in seen):
            result,md=allocation(allocation_run,(allocation_run/'COMPLETE.json').exists())
            save(allocation_run/'monitor/AUDITED_PROGRESS',result,md);last_allocation=signature
            if allocation_complete:
                seen.add('allocation_final')
                print('ALLOCATION_COMPLETED_AND_AUDITED',flush=True)
        if allocation_complete and (phase_run/'COMPLETE.json').exists():
            print('ALL_FOLLOWUP_STUDIES_COMPLETED_AND_AUDITED',flush=True)
            return
        time.sleep(15)
    print('MONITOR_TIME_LIMIT_REACHED; workers were not interrupted',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',type=Path,required=True)
    parser.add_argument('--allocation',type=Path,required=True);args=parser.parse_args()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2] or cpus));os.nice(10)
    main(args.phase,args.allocation)
