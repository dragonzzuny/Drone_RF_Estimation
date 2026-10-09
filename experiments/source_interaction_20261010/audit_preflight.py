"""Recompute all saved TRAIN4 score summaries without new inference."""
import argparse
import math
from pathlib import Path
import statistics
import run_preflight as worker


def main(study,public):
    w=worker.watch;p=w.read(study/'PROTOCOL.json');worker.verify(study,p)
    complete=w.read(study/'COMPLETE.json')
    if complete['protocol_sha256']!=w.digest(study/'PROTOCOL.json'):
        raise ValueError('Wrong completed preflight')
    if [r['arm'] for r in complete['results']]!=list(worker.ARMS):
        raise ValueError('Missing or duplicated arm')
    starts=[];rows=[]
    for result in complete['results']:
        arm=result['arm']
        if result!=w.read(study/f'{arm}_COMPLETE.json') or result['updates']!=32:
            raise ValueError('Completion receipt differs')
        if result['parameters']!=p['parameters'][arm]:raise ValueError('Capacity differs')
        if [h['step'] for h in result['history']]!=[0,1,8,16,32]:raise ValueError('Missing fit stage')
        starts.append(result['history'][0]['rows'])
        for h in result['history']:
            if [r['case'] for r in h['rows']]!=[0,1,2,3]:raise ValueError('Wrong TRAIN identities')
            for count in (2,3):
                group=[r for r in h['rows'] if r['count']==count]
                if len(group)!=2 or any(len(r['nmse'])!=count or len(r['si_sdr'])!=count for r in group):
                    raise ValueError('Wrong number of source scores')
                expected=next(r for r in h['by_count'] if r['count']==count)
                nmse=statistics.mean(v for r in group for v in r['nmse'])
                si=statistics.mean(v for r in group for v in r['si_sdr'])
                if not w.close(nmse,expected['mean_nmse']) or not w.close(si,expected['mean_si_sdr']):
                    raise ValueError('Stored mean differs from source scores')
                if not math.isfinite(nmse) or not math.isfinite(si):raise ValueError('Nonfinite score')
                rows.append(dict(arm=arm,step=h['step'],count=count,nmse=nmse,si=si))
        actual=all(b['mean_nmse']<a['mean_nmse'] for a,b in zip(result['history'][0]['by_count'],result['history'][-1]['by_count']))
        if actual!=result['both_counts_improved']:raise ValueError('Wrong fit success flag')
    if starts[0]!=starts[1] or not complete['initial_gpu_predictions_exactly_equal']:
        raise ValueError('Initial conditions differ')
    audited=dict(status='PASS',source_sha256=w.digest(Path(__file__)),
        protocol_sha256=w.digest(study/'PROTOCOL.json'),complete_sha256=w.digest(study/'COMPLETE.json'),
        all_stages_and_components_recomputed=True,initial_saved_scores_exactly_equal=True,
        source_and_capacity_checks=True,model_updates=0,waveform_reads=0,heldout_read=False)
    w.write(public.with_suffix('.json'),audited)
    lines=['# 성분 상호작용 출력층: 원 규모 GPU TRAIN4 검사','',
        '같은 보존 U-Net e2·학습 혼합4개·각32업데이트·원 손실·AdamW1e-4·동일 초기 예측이다. '
        '본체는 양쪽 모두 학습했다. 아래 수치는 고정 TRAIN 적합도이며 개발검증 성능이 아니다. '
        '검사 가중치는 폐기했다. 모든 기록 단계와 성분 수를 함께 보고한다.','',
        '| 군 | 업데이트 | 성분 수 | TRAIN NMSE ↓ | TRAIN 복소 SI-SDR ↑ dB |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['arm']}|{r['step']}|{r['count']}|{r['nmse']:.6f}|{r['si']:.3f}|")
    lines+=['','두 군의 초기 GPU 파형·개수 logits가 정확히 같았고, 추가층의 내부 attention에도 '
        '첫 단계 이후 기울기가 전달됐다. 통과 여부는 실행·최적화 조건에 한정하며 새 기록 일반화의 증거가 아니다.','']
    for r in complete['results']:
        lines.append(f"- {r['arm']}: {r['parameters']:,}파라미터, {r['seconds']:.1f}초, "
                     f"최대 할당 {r['peak_bytes']/2**30:.3f}GiB, 두 성분 수의 초기 대비 NMSE 감소={r['both_counts_improved']}.")
    w.write(public.with_suffix('.md'),'\n'.join(lines)+'\n')
    print(audited)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();main(a.study,a.public)
