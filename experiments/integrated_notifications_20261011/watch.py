"""Deliver saved integrated-model epochs to the user's local GNOME desktop."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('epoch_notify',ROOT/'experiments/epoch_notifications_20261010/notify_epochs.py')
notify=importlib.util.module_from_spec(spec);spec.loader.exec_module(notify)


def run(folder, output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=output/'STATE.json'
        state=notify.read(saved) if saved.exists() else dict(seen=0,events=[])
        while True:
            try:
                for path in sorted(folder.glob('EPOCH_*.json')):
                    epoch=int(path.stem.split('_')[-1])
                    if epoch<=state['seen']:continue
                    assert epoch==state['seen']+1
                    event=notify.read(path)
                    assert notify.digest(folder/f'VALIDATION_{epoch:03d}.json')==event['validation_sha256']
                    prev=folder/f'EPOCH_{epoch-1:03d}.json'
                    title,body=notify.render('통합 RF U-Net',event,notify.read(prev) if prev.exists() else None)
                    receipt=notify.send(title,body)
                    state['events'].append(dict(epoch=epoch,notification_id=receipt,title=title,body=body,
                        epoch_sha256=notify.digest(path),delivery='LOCAL_DESKTOP_SERVER_ACKNOWLEDGED',
                        human_seen=False,time=time.time()))
                    state['seen']=epoch
                    notify.write(saved,state)
                status='WATCHING'
                complete=folder/'COMPLETE.json'
                if complete.exists() and state['seen']==notify.read(complete)['epochs']:
                    status='COMPLETED_RUN_REPORTED'
                if (folder/'FAILURE.json').exists():status='TRAINING_FAILURE_RECORDED'
                state.update(status=status,pid=os.getpid(),time=time.time(),training_run=str(folder))
                notify.write(saved,state)
                if status!='WATCHING':return
            except Exception as error:
                notify.write(output/'ERROR.json',dict(error=repr(error),time=time.time()))
            time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    run(a.training_run.resolve(),a.output.resolve())
