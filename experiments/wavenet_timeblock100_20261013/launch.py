"""Prepare CPU features, register the fixed experiment, then run on the GPU."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'local/wavenet_timeblock100_20261013_v1'


def main():
    assert not (OUT/'PROTOCOL.json').exists(), 'Use train.py --resume for a registered run'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    subprocess.run(['taskset','-c','12,13',sys.executable,str(HERE/'prepare.py')],env=env,check=True)
    subprocess.run([sys.executable,str(HERE/'train.py'),'--register'],check=True)
    os.execv(sys.executable,[sys.executable,str(HERE/'train.py')])


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        result=dict(status='FAILED_STARTUP',pid=os.getpid(),time=time.time(),error=traceback.format_exc())
        (OUT/'STARTUP_FAILURE.json').write_text(json.dumps(result,indent=2)+'\n')
        (OUT/'STATE.json').write_text(json.dumps(result,indent=2)+'\n')
        raise
