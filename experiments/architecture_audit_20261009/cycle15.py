"""One-variable WaveNet dilation comparison motivated by TRAIN lag diagnostics.

Same 30 layers, 128 channels, parameter values, 63872-sample input and 32-update
TRAIN fitting protocol as the cycle10 check. Does not supply 20ms complex I/Q.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

import fit_diagnostic as reference
from native_wavenet import build_native_wavenet
from native_data import sha256, write_json


def build_cycle15():
    model = build_native_wavenet()
    for index, block in enumerate(model.blocks):
        dilation = 2**(index%15)
        block.filter_gate.dilation = (dilation,)
        block.filter_gate.padding = (dilation,)
    return model


def cpu_check():
    torch.set_num_threads(2)
    first, second = build_native_wavenet(), build_cycle15()
    assert sum(p.numel() for p in second.parameters())==4601867
    for key,value in first.state_dict().items():
        assert torch.equal(value,second.state_dict()[key]),key
    receptive = 1+sum(2*m.filter_gate.dilation[0] for m in second.blocks)
    assert receptive==131069
    del first,second
    gc.collect()
    return dict(status='PASS',identical_initial_parameter_tensors=True,parameters=4601867,
        layers=30,channels=128,cycle=15,receptive_field_samples=receptive,receptive_field_ms=receptive/100_000,
        radius_samples=65534,input_samples=63872,input_ms=.63872,note='finite input caps available waveform context; convolution padding adds no observations')


def run(root, prior):
    if (root/'PROTOCOL.json').exists():raise ValueError('Refuse duplicate registration')
    root.mkdir(parents=True,exist_ok=True)
    protocol=reference.read(prior/'PROTOCOL.json')
    protocol.update(status='REGISTERED_SINGLE_VARIABLE_CYCLE15_TRAIN_FIT',
        arms={'wavenet_native':4601867},architecture_variant='30x128 noncausal WaveNet with cycle15 instead of cycle10',
        first_diagnostic=str(prior),first_diagnostic_protocol_sha256=sha256(prior/'PROTOCOL.json'),
        change='only Conv1d dilation and symmetric padding; same tensor initial values, data, loss, optimizer, steps',
        motivation='TRAIN-only exploratory repeated complex lags near42813 and50000 exceed the cycle10 convolution path',
        convolutional_receptive_field_samples=131069,input_context_unchanged=True)
    protocol['source_sha256'][str(Path(__file__).relative_to(reference.ROOT))]=sha256(Path(__file__))
    for rel,digest in protocol['source_sha256'].items():
        if sha256(reference.ROOT/rel)!=digest:raise ValueError('Referenced source changed')
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(reference.ROOT/rel,dest)
    write_json(root/'PROTOCOL.json',protocol)
    write_json(root/'CPU_CHECK.json',cpu_check())
    begin=time.time()
    while not (prior/'COMPLETE.json').exists():
        if (prior/'FAILURE.json').exists():raise RuntimeError('Prior full-input diagnostic failed')
        if time.time()-begin>7200:raise TimeoutError('Prior diagnostic wait exceeded')
        write_json(root/'STATE.json',dict(status='WAITING_FOR_CYCLE10_AND_UNET_DIAGNOSTIC',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    if sha256(prior/'PROTOCOL.json')!=protocol['first_diagnostic_protocol_sha256']:
        raise ValueError('Matched diagnostic protocol changed')
    # Explicit adaptation of the frozen diagnostic runner in this process only.
    reference.build_native_wavenet=build_cycle15
    reference.run(root,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve()
    try:
        run(root,args.prior.resolve())
    except Exception:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
        raise
