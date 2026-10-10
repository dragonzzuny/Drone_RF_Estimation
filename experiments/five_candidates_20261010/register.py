"""Seal the user-authorized five-candidate execution amendment."""
import shutil
import time
from workflow import ROOT, PUBLIC, RUN, SEP, ORDER, SPECS, read, write, digest

CHECKS = {
 'frozen_tf': ['TF_AXIS_CPU_CHECK.json','TF_AXIS_ADAPTATION_CPU_CHECK.json','TF_AXIS_ADAPTATION_AUDIT.json'],
 'ordered_context': ['ORDERED_CONTEXT_CHECK.json','ORDERED_CONTEXT_AUDIT.json'],
 'balanced_head': ['BALANCED_SOURCE_HEAD_CPU_CHECK.json'],
 'wave_guard': ['WAVE_UPDATE_GUARD_CPU_CHECK.json','WAVE_UPDATE_GUARD_FULL_CHECK.json','WAVE_UPDATE_GUARD_FINAL_AUDIT.json'],
}


def register():
    assert not (RUN/'PROTOCOL.json').exists(),'Already sealed'
    RUN.mkdir(parents=True,exist_ok=True)
    sources={};pins={};candidates={}
    guard=read(ROOT/'local/wave_update_guard_20261010_v1/PROTOCOL.json')
    for name,spec in SPECS.items():
        source_root=ROOT/'local'/(spec['origin'] or 'balanced_head_probe_20261010_v1')
        prior=read(source_root/'PROTOCOL.json')
        for rel,sha in prior['source_sha256'].items():
            assert digest(ROOT/rel)==sha,rel
            if rel in sources: assert sources[rel]==sha,rel
            sources[rel]=sha
        checks=CHECKS[name]
        for f in checks:
            data=read(PUBLIC/f);assert data['status']=='PASS',f
            pins[str(PUBLIC/f)]=digest(PUBLIC/f)
            for rel,sha in data.get('source_sha256',{}).items():
                assert digest(ROOT/rel)==sha,rel
                if rel in sources: assert sources[rel]==sha,rel
                sources[rel]=sha
        folder=RUN/name;folder.mkdir(exist_ok=True)
        candidates[name]=dict(spec,checks=checks,imported_epochs=int(bool(spec['origin'])))
        pins[str(source_root/'PROTOCOL.json')]=digest(source_root/'PROTOCOL.json')
        if spec['origin']:
            origin=source_root/spec['subdir'];candidates[name]['origin_protocol_sha256']=digest(source_root/'PROTOCOL.json')
            for f in ('LAST.pt','BEST.pt','EPOCH_001.json','VALIDATION_000.json','VALIDATION_001.json'):
                path=origin/f;assert path.exists(),str(path)
                pins[str(path)]=digest(path)
                # Independent copies protect resume state from later origin cleanup.
                shutil.copyfile(path,folder/('RESUME_INPUT.pt' if f=='LAST.pt' else f))
            receipt=read(folder/'EPOCH_001.json')
            receipt.update(imported=True,original_event_sha256=digest(origin/'EPOCH_001.json'),
                           candidate=name,schedule_epoch=spec['schedule_start'])
            write(folder/'EPOCH_001.json',receipt)
    for f in PathHere.glob('*.py'): sources[str(f.relative_to(ROOT))]=digest(f)
    v=ROOT/'experiments/recursive_20261010/validation.py';sources[str(v.relative_to(ROOT))]=digest(v)
    plan=PUBLIC/'FIVE_CANDIDATES_PLAN_KO.md';pins[str(plan)]=digest(plan)
    for path in (PathSepProtocol,SEP/'EXECUTION_PRIORITY.json',ROOT/guard['parent_checkpoint'],
                 ROOT/guard['baseline'],ROOT/guard['preparation']/'PREPARATION.json'):
        pins[str(path)]=digest(path)
    for path in (ROOT/guard['preparation']/'features').glob('*pack_*.json'):
        pins[str(path)]=digest(path)
    for rel,sha in sources.items():
        dest=RUN/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest);assert digest(dest)==sha
    p=dict(status='REGISTERED_FIVE_CANDIDATES_50',order=list(ORDER),candidates=candidates,
        septda_run=str(SEP),septda_execution_amendment_sha256=digest(SEP/'EXECUTION_PRIORITY.json'),
        parent_checkpoint=guard['parent_checkpoint'],parent_checkpoint_sha256=guard['parent_checkpoint_sha256'],
        baseline=guard['baseline'],preparation=guard['preparation'],gpu_display_policy=guard['gpu_display_policy'],
        source_sha256=sources,pinned_files=pins,maximum_epochs=50,updates_per_epoch=75,examples_per_epoch=2400,
        fixed_mixture_count=12000,effective_batch=32,microbatch=1,seed=0,precision='FP32 TF32off',
        schedule='Each original phase retained; all five fixed schedules each used ten times by block50',
        lr_rule='Preserve imported optimizer rates; monitor starts at20; halve all group rates after10 nonimprovements >=1e-5',
        selection='Minimum DEV mean NMSE counts2/3 including e0; keep actual and selected results separate',
        control='Reuse completed common control only; no new control runs or matched50 superiority claims',
        training='Full architecture and original objective/update method of each candidate; fresh balanced head from common parent',
        milestones=[5,10,20,30,40,50],early_stopping=False,heldout_read=False,independent_test=False,
        scope='RFUAV TRAIN8/DEV5 same original band and native100MS/s; ambiguous transmitter provenance retained',
        failure='Stop queue on failed worker/integrity/numerical checks; retain evidence',
        user_authorization='Latest user: run the five selected candidates sequentially for50each and keep reporting',
        registered_at=time.time())
    write(RUN/'PROTOCOL.json',p)
    write(RUN/'STATE.json',dict(status='REGISTERED_PENDING_PREFLIGHT',time=time.time()))
    print(dict(status=p['status'],source_files=len(sources),pinned_files=len(pins)),flush=True)


PathHere=ROOT/'experiments/five_candidates_20261010'
PathSepProtocol=SEP/'PROTOCOL.json'
if __name__=='__main__': register()
