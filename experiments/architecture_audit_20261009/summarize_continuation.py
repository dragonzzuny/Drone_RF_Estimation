"""Audited continuation summary with all matched-epoch LR comparisons."""
import argparse
from pathlib import Path
import summarize_loss_study as base_summary
import watch_epochs as watch


def main(study, reference, audit_path, output):
    p, old = watch.read(study/'PROTOCOL.json'), watch.read(reference/'PROTOCOL.json')
    if p['executed_epochs'] != [2, 3] or p['imported_updates_per_arm'] != 75:
        raise ValueError('Unexpected continuation protocol')
    for key in ('parent_checkpoint_sha256','preparation_sha256','parameters','seed',
                'effective_batch','microbatch','validation_cases','examples_per_epoch',
                'train_schedule_epochs','epochs_per_arm'):
        if p[key] != old[key]:
            raise ValueError('Unmatched reference: '+key)
    for rel in set(p['source_sha256']) & set(old['source_sha256']):
        if p['source_sha256'][rel] != old['source_sha256'][rel]:
            raise ValueError('Shared implementation changed: '+rel)
    base_summary.summarize(study,audit_path,output)
    result=watch.read(output.with_suffix('.json'))
    _,identities=watch.validate(Path(p['validation_identity_template']),None)
    comparison=[]
    for epoch in (1,2,3):
        for arm in p['arms']:
            path=reference/arm/f'VALIDATION_{epoch:03d}.json'
            before,_=watch.validate(path,identities)
            after=result['histories'][arm][epoch]
            result['validation_sha256'][f'reference/{arm}/e{epoch}']=watch.digest(path)
            for count in (2,3):
                a=next(r for r in before['by_count'] if r['count']==count)
                b=next(r for r in after['by_count'] if r['count']==count)
                comparison.append(dict(arm=arm,epoch=epoch,count=count,
                    nmse_high_lr=a['mean_nmse'],nmse_low_lr=b['mean_nmse'],
                    si_high_lr=a['mean_si_sdr'],si_low_lr=b['mean_si_sdr'],
                    both_better=b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr']))
    result.update(source_sha256=watch.digest(Path(__file__)),
        base_summary_code_sha256=watch.digest(Path(base_summary.__file__)),
        same_actual_epoch_learning_rate_comparison=comparison,
        imported_updates_per_arm=75,executed_updates_per_arm=150,
        total_fine_tune_updates_per_arm=225,adaptive_development_extension=True)
    result.pop('same_actual_e1_learning_rate_comparison')
    watch.write(output.with_suffix('.json'),result)
    lines=output.with_suffix('.md').read_text().splitlines()
    lines[0]='# 낮은 학습률 U-Net 연속 비교: 총 3 epoch 완료 결과'
    lines.insert(4,'각 군의 첫 75업데이트는 앞선 실험에서 가져왔으며 재학습하지 않았다. '
        '모델·AdamW 상태·RNG를 그대로 이어 이번 실행에서 각 150업데이트만 추가했다. '
        '처음 시작할 때만 새 optimizer를 사용했다. 첫 결과를 본 뒤 정한 개발 연장이다.')
    lines+=['','## 학습률 5e-4와 1e-4: 같은 실제 epoch끼리 비교','',
        '| 손실 | 추가 epoch | 성분 수 | NMSE 5e-4 → 1e-4 ↓ | SI-SDR 5e-4 → 1e-4 ↑ dB | 두 지표 개선 |',
        '|---|---:|---:|---|---|---|']
    for r in comparison:
        lines.append(f"| {r['arm']} | {r['epoch']} | {r['count']} | "
            f"{r['nmse_high_lr']:.6f} → {r['nmse_low_lr']:.6f} | "
            f"{r['si_high_lr']:.3f} → {r['si_low_lr']:.3f} | {r['both_better']} |")
    lines+=['','세 epoch와 모든 실패 조건을 포함했다. 최종 선택은 원래의 raw NMSE 규칙을 유지하며, '
        '낮은 학습률이 높은 학습률보다 좋다는 관찰과 시작 모델을 개선했다는 주장은 구분한다.','']
    watch.write(output.with_suffix('.md'),'\n'.join(lines))
    print({'status':'COMPLETE','matched_comparisons':len(comparison),
           'selected_jointly_improves_parent':result['selected_jointly_improves_parent']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','reference','audit','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    main(a.study.resolve(),a.reference.resolve(),a.audit.resolve(),a.output.resolve())
