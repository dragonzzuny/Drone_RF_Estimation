"""CPU TRAIN48 fit and inference-time branch bypass after the first RF epoch."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import numpy as np
import torch
import train as t
from drone_rf.waveform import waveform_metrics

w = t.w


def batch(raw):
    return {k: torch.as_tensor(np.asarray(raw[k])[None]) for k in
            ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')}


@torch.inference_mode()
def evaluate(net, item, index):
    output, logits = t.worker.predict(net, item)
    m = waveform_metrics(output, item['references'], item['active'], item['mixture'])
    n = int(item['construction_count'][0])
    power = m['reference_power'][0, :n].tolist()
    row = dict(index=int(index), count=n, nmse=m['nmse'][0, :n].tolist(),
               si_sdr=m['si_sdr'][0, :n].tolist(), reference_power=power,
               weakest_index=int(np.argmin(power)), predicted_count=int(logits.argmax(-1)) + 1,
               sum_relative_error=float(m['sum_relative_error'][0]))
    assert np.isfinite(row['nmse'] + row['si_sdr']).all() and row['sum_relative_error'] < 1e-9
    return row, output


def summarize(rows):
    result = []
    for model in sorted({r['model'] for r in rows}):
        for count in (1, 2, 3):
            part = [r for r in rows if r['model'] == model and r['count'] == count]
            result.append(dict(model=model, count=count, cases=len(part),
                mean_nmse=statistics.mean(v for r in part for v in r['nmse']),
                mean_si_sdr=statistics.mean(v for r in part for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in part)))
    return result


def run(root, study):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Already registered')
    p = w.read(study / 'PROTOCOL.json'); t.verify(study, p)
    audit = w.read(t.PUBLIC / 'SEPTDA_RF_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['complete_sha256'] == w.digest(study / 'COMPLETE.json')
    prior_path = t.PUBLIC / 'SOURCE_INTERACTION_TRAIN_FIT.json'
    prior = w.read(prior_path)
    prior_audit = w.read(t.PUBLIC / 'SOURCE_INTERACTION_TRAIN_FIT_AUDIT.json')
    assert prior_audit['status'] == 'PASS' and prior_audit['completed_diagnosis_sha256'] == w.digest(prior_path)
    assert prior['checkpoint_validation_sha256']['parent/e0']['checkpoint'] == p['original_protocol']['parent_checkpoint_sha256']
    fresh = Path(p['comparison_control']).parent
    files = [prior_path, study / 'PROTOCOL.json', study / 'ACTUAL_001.pt', study / 'INITIAL.pt',
             study / 'VALIDATION_001.json', fresh / 'ACTUAL_001.pt', fresh / 'VALIDATION_001.json', Path(__file__)]
    pins = {str(f): w.digest(f) for f in files}
    protocol = dict(status='REGISTERED_CPU_TRAIN48_BRANCH_DIAGNOSIS',
        indices=prior['train_indices'], train_schedule_epoch=1, pinned_files=pins,
        study_protocol_sha256=w.digest(study / 'PROTOCOL.json'), new_model_updates=0,
        heldout_read=False, selection='Reuse the prior fixed TRAIN48 without selecting by new results',
        interpretation='Branch bypass at inference on the jointly trained e1 backbone; not an independently trained architecture ablation.',
        registered_at=time.time())
    snapshot = root / 'source_snapshot' / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    w.write(root / 'PROTOCOL.json', protocol)
    torch.set_num_threads(2)
    train = t.worker.NativeMixtures(p['original_protocol']['preparation'], 'train_pack', 1)
    dev = t.worker.NativeMixtures(p['original_protocol']['preparation'], 'validation_pack', 1)
    rows = [r for r in prior['rows'] if r['model'] == 'parent/e0']
    parent = {r['index']: r for r in rows}; assert len(parent) == 48
    backend, corrections = [], []
    done = 0; started = time.time()
    for label, folder, augmented in [('same_schedule_control', fresh, False), ('septda_e1', study, True)]:
        saved = torch.load(folder / 'ACTUAL_001.pt', map_location='cpu', weights_only=False)
        assert saved['epoch'] == 1 and saved['updates'] == 75
        net = t.make_net() if augmented else t.worker.make_model('retained_unet')
        net.load_state_dict(saved['model']); del saved
        net.eval()
        gpu_rows = w.read(folder / 'VALIDATION_001.json')['rows']
        for count in (1, 2, 3):
            expected = next(r for r in gpu_rows if r['count'] == count)
            actual, _ = evaluate(net, batch(dev[expected['index']]), expected['index'])
            dn = max(abs(a - b) for a, b in zip(actual['nmse'], expected['nmse']))
            ds = max(abs(a - b) for a, b in zip(actual['si_sdr'], expected['si_sdr']))
            assert dn < 2e-5 and ds < 2e-3 and actual['predicted_count'] == expected['predicted_count']
            backend.append(dict(model=label, index=expected['index'], nmse_max_error=dn, si_sdr_max_error=ds))
        for index in protocol['indices']:
            item = batch(train[index]); row, full = evaluate(net, item, index)
            assert row['count'] == parent[index]['count']
            assert np.allclose(row['reference_power'], parent[index]['reference_power'], rtol=1e-7, atol=1e-12)
            rows.append(dict(model=label, **row)); done += 1
            if augmented:
                original_forward = net._forward
                try:
                    net._forward = net._septda_parent_forward
                    bypass, disabled = evaluate(net, item, index)
                finally:
                    net._forward = original_forward
                rows.append(dict(model='septda_branch_bypassed', **bypass)); done += 1
                assert bypass['predicted_count'] == row['predicted_count']
                power = item['mixture'].abs().square().sum().item()
                ratio = (full - disabled).abs().square().sum().item() / power
                corrections.append(dict(index=index, count=row['count'],
                    summed_four_output_change_energy_over_mixture_energy=ratio))
            w.write(root / 'PARTIAL.json', dict(rows=rows, backend_checks=backend, corrections=corrections))
            w.write(root / 'STATE.json', dict(status='CPU_TRAIN48_DIAGNOSIS', model=label,
                    cases=done, total=144, pid=os.getpid(), seconds=time.time()-started, time=time.time()))
        del net; gc.collect()
    for path, sha in pins.items():
        assert w.digest(Path(path)) == sha
    assert w.digest(snapshot) == pins[str(Path(__file__))]
    t.verify(study, p)
    assert len(rows) == 192 and done == 144 and len(backend) == 6 and len(corrections) == 48
    assert not torch.cuda.is_initialized()
    result = dict(status='COMPLETE_CHECKED', protocol_sha256=w.digest(root / 'PROTOCOL.json'),
                  rows=rows, summaries=summarize(rows), backend_checks=backend, corrections=corrections,
                  seconds=time.time()-started, new_model_updates=0, cuda_initialized=False,
                  heldout_read=False, independent_test=False,
                  limitation='Small reused TRAIN48; different activity/power distribution from DEV, not a pure generalization-gap estimate. Inference bypass is not a trained ablation. Prior parent results reused after hash and reference-power checks.')
    w.write(root / 'COMPLETE.json', result)
    w.write(t.PUBLIC / 'SEPTDA_RF_TRAIN_DIAGNOSIS.json', result)
    lines = ['# SepTDA 참고 분리부: 학습 자료와 분리 경로 진단', '',
             '새 후보의 첫 epoch 후 기존 고정 TRAIN48을 다시 평가했다. 부모의 기존 검산 결과를 재사용하고, '
             '같은 일정 대조·새 모델·새 경로를 추론 때만 끈 모델을 CPU에서 비교했다. '
             '새 학습 0회, 보류 자료 접근 0회이며 원 규모 모델·창 길이를 유지했다.', '',
             '| 조건 | 성분 수 | 혼합 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 최약 NMSE ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for r in result['summaries']:
        lines.append(f"|{r['model']}|{r['count']}|{r['cases']}|{r['mean_nmse']:.6f}|{r['mean_si_sdr']:.3f}|{r['weakest_nmse']:.6f}|")
    lines += ['', '경로를 끈 조건은 공동 학습된 본체를 그대로 사용한다. 별도로 학습한 구조 제거 대조가 아니며 '
              'TRAIN48과 DEV630의 전력·활동 분포가 달라 두 점수 차이만으로 과적합 정도를 산정하지 않는다. '
              '경로 기여량은 같은 출력 번호에서 4출력 변화 에너지 합 / 혼합 에너지로 계산하며 분리 정확도가 아니다. '
              '각 체크포인트의 CPU/GPU 검증 사례를 개수별 1개씩 재현했다.', '',
              '[전체 사례·백엔드 검사·경로 기여량](SEPTDA_RF_TRAIN_DIAGNOSIS.json)']
    w.write(t.PUBLIC / 'SEPTDA_RF_TRAIN_DIAGNOSIS_KO.md', '\n'.join(lines)+'\n')
    w.write(root / 'STATE.json', dict(status='COMPLETE_CHECKED', cases=144, pid=os.getpid(), time=time.time()))
    print(dict(status=result['status'], seconds=result['seconds'], summaries=result['summaries']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args(); root = args.run.resolve()
    try:
        run(root, args.study.resolve())
    except Exception:
        w.write(root / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
