"""User amendment: stop at the next completed epoch without joint progress."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.review')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def comparison(actual,incumbent):
    checks=[]
    for a,b in zip(actual['by_count'][1:],incumbent['by_count'][1:]):
        assert a['count']==b['count']
        checks.append(dict(count=a['count'],nmse_better=a['mean_nmse']<b['mean_nmse'],
            si_better=a['mean_si_sdr'] is not None and b['mean_si_sdr'] is not None and a['mean_si_sdr']>b['mean_si_sdr'],
            weakest_nonworse=a['weakest_nmse']<=b['weakest_nmse']))
    return all(c[k] for c in checks for k in ('nmse_better','si_better','weakest_nonworse')),checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True);a=parser.parse_args();root=a.run.resolve();pid=a.pid
    proc=Path('/proc',str(pid))
    expected=str(Path(__file__).resolve().parents[2]/'experiments/integrated_20261011/train.py').encode()
    command=(proc/'cmdline').read_bytes().split(b'\0')
    assert expected in command and str(root).encode() in command
    birth=(proc/'stat').read_text().rsplit(') ',1)[1].split()[19]
    protocol=read(root/'PROTOCOL.json');incumbent=read(protocol['baseline']);selected=0;epoch=1
    amendment=root/'USER_EPOCH_REVIEW.json'
    if amendment.exists():raise RuntimeError('Observer already registered')
    write(amendment,dict(status='USER_OVERRIDE_MINIMUM_FIVE',source_sha256=sha(__file__),
        protocol_sha256=sha(root/'PROTOCOL.json'),training_pid=pid,process_start_ticks=birth,
        rule='Every next completed epoch must strictly improve counts2/3 NMSE and complex SI-SDR, with weakest NMSE nonworse, versus last jointly accepted model (initially parent). Otherwise stop saved trajectory.',
        user_authorization='다음 epoch도 개선 안 되면 중지하고 다음 실험',registered_at=time.time()))
    while proc.exists():
        eventpath=root/f'EPOCH_{epoch:03d}.json'
        if not eventpath.exists():time.sleep(.5);continue
        event=read(eventpath)
        assert event['epoch']==epoch and event['updates']==75*epoch
        assert event['protocol_sha256']==sha(root/'PROTOCOL.json')
        assert sha(root/f'VALIDATION_{epoch:03d}.json')==event['validation_sha256']
        improved,checks=comparison(event['validation'],incumbent)
        decision=dict(epoch=epoch,incumbent_epoch=selected,improved=improved,checks=checks,time=time.time())
        write(root/f'USER_REVIEW_{epoch:03d}.json',decision)
        if improved:
            selected=epoch;incumbent=event['validation'];epoch+=1;continue
        assert (proc/'stat').read_text().rsplit(') ',1)[1].split()[19]==birth
        assert expected in (proc/'cmdline').read_bytes().split(b'\0')
        before=read(root/'STATE.json')
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not proc.exists():break
            time.sleep(.1)
        assert not proc.exists(),'Worker did not stop; no completion receipt written'
        assert sha(root/'LAST.pt')==event['last_sha256']
        receipt=dict(status='STOPPED_AT_USER_EPOCH_REVIEW',reason='NO_JOINT_IMPROVEMENT',
            epochs=epoch,updates=75*epoch,best=event['best'],incumbent_joint_epoch=selected,
            last_state_before_stop=before,checkpoint_sha256_verified=True,
            protocol_sha256=sha(root/'PROTOCOL.json'),amendment_sha256=sha(amendment),
            time=time.time(),heldout_read=False,old_maximum_budget_completed=False)
        write(root/'COMPLETE.json',receipt)
        write(root/'STATE.json',dict(status=receipt['status'],epoch=epoch,pid=None,time=time.time()))
        public=Path(__file__).resolve().parents[2]/'reports/2026-10-11'
        write(public/'INTEGRATED_USER_REVIEW.json',dict(decision=decision,stop=receipt,amendment=read(amendment)))
        progress=public/'INTEGRATED_PROGRESS.json'
        value=read(progress);value['status']=receipt['status'];value['user_review']=decision;write(progress,value)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
        return
    write(root/'USER_REVIEW_OBSERVER_EXIT.json',dict(reason='Worker exited before next receipt',next_epoch=epoch,time=time.time()))


if __name__=='__main__':main()
