"""Paired success/adverse examples, same waveform scale and full time span.

Case7 was the predeclared equal-power triple; case4 is a disclosed adverse
weak-FPV development case. References align output labels ONCE per record only.
No local oracle routing or reference gain/phase fitting enters these figures.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import fft
from evaluate_parent_full_context import ROOT,read,sha
from sequence_data import Sequences,ROLES,LENGTH

SOURCE=ROOT/'local/spectral_routing_24_20261013_v1'
ROUTING=ROOT/'local/constrained_routing_20261013_v1'
DEST=ROOT/'reports/2026-10-13/figures'
FFT=1024
FS=100000000


def power_image(x):
    assert x.shape==(LENGTH,) and LENGTH%FFT==0
    window=np.sqrt(.5-.5*np.cos(2*np.pi*np.arange(FFT)/FFT)).astype(np.float32)
    frames=x.reshape(-1,FFT);power=np.empty((len(frames),FFT),np.float64)
    for start in range(0,len(frames),64):
        z=fft.fft(frames[start:start+64]*window,axis=-1,workers=1)
        power[start:start+64]=(z.real.astype(np.float64)**2+z.imag.astype(np.float64)**2)/(FFT*np.sum(window.astype(np.float64)**2))
    power=fft.fftshift(power,axes=-1)
    assert power.shape==(2040,1024)
    # Exactly2 frequency bins and2 nonoverlapping STFT frames per display pixel.
    # Average LINEAR powers before dB; identical operation for every panel.
    return power.reshape(1020,2,512,2).mean((1,3)).T


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=read(SOURCE/'RESULT.json');comparison=read(ROUTING/'RESULT.json')
    assert sha(SOURCE/'RESULT.json')==read(SOURCE/'AUDIT.json')['result_sha256']
    assert sha(ROUTING/'RESULT.json')==read(ROUTING/'AUDIT.json')['result_sha256']
    data=Sequences(ROLES[1])
    for index in (7,4):
        case=next(r for r in result['cases'] if r['index']==index)
        metrics=next(r for r in comparison['groups']['B24']['cases'] if r['index']==index)
        path=SOURCE/f'OUTPUT_{index:03d}.npy'
        if case['reused_parent_output']:path=ROOT/'local/full_context_parent_20261013_v1'/path.name
        assert sha(path)==case['parent_output_sha256']
        order_path=ROUTING/'B24'/f'ORDERS_{index:03d}.json'
        assert sha(order_path)==metrics['orders_sha256']
        parent=np.load(path,mmap_mode='r',allow_pickle=False);refs=data.components(index,True);mix=refs.sum(0)
        routed=np.empty_like(parent)
        for order in read(order_path)['orders']:
            a,b=order['start'],order['stop'];routed[:,a:b]=parent[[*order['permutation'],3],a:b]
        assert np.array_equal(parent[3],routed[3])
        mixture_power=float(np.mean(abs(mix.astype(np.complex128))**2))
        sources=[('mixture',mix)]
        for j in range(3):
            sources.extend([(f'ref_{j}',refs[j]),
                (f'parent_{j}',parent[metrics['parent']['assignment'][j]]),
                (f'routed_{j}',routed[metrics['candidate']['assignment'][j]])])
        sources.extend([('parent_background',parent[3]),('routed_background',routed[3])])
        power={key:power_image(value)/mixture_power for key,value in sources}
        # One disclosed display range for BOTH cases and every panel. Preserve
        # clipping fractions; this range does not alter any waveform metrics.
        lower,upper=-75.,5.
        db={key:10*np.log10(np.maximum(p,np.finfo(np.float64).tiny)) for key,p in power.items()}
        clipping={key:dict(below=float(np.mean(value<lower)),above=float(np.mean(value>upper))) for key,value in db.items()}
        prefix=DEST/f'stft_B{index:02d}';files=[prefix.with_suffix(x) for x in ('.pdf','.svg','.png','.json','.md')]
        assert not any(p.exists() for p in files)
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                             'svg.fonttype':'none','axes.titlesize':10,'axes.labelsize':9}):
            fig=plt.figure(figsize=(12,11),layout='constrained');grid=fig.add_gridspec(5,3,height_ratios=[.8,1,1,1,.8])
            axes=[]
            def panel(position,key,title,ylabel=False):
                ax=fig.add_subplot(position);axes.append(ax)
                image=ax.imshow(db[key],origin='lower',aspect='auto',interpolation='nearest',cmap='cividis',
                    vmin=lower,vmax=upper,extent=[0,LENGTH/FS*1000,-FS/2e6,FS/2e6],rasterized=True)
                ax.set_title(title,loc='left');ax.set_xlabel('Time (ms)')
                if ylabel:ax.set_ylabel('Frequency offset (MHz)')
                ax.set_yticks([-40,-20,0,20,40]);return ax,image
            _,image=panel(grid[0,:],'mixture',f'B{index:02d}: observed mixture, full20.8896ms; nominal levels {case["nominal_levels_db"]} dB',True)
            for j,category in enumerate(case['categories']):
                cat=category.replace('DJI ','')
                panel(grid[j+1,0],f'ref_{j}',cat+' — reference (evaluation only)',True)
                panel(grid[j+1,1],f'parent_{j}',f'U-Net + C8: NMSE={metrics["parent"]["nmse"][j]:.3f}')
                panel(grid[j+1,2],f'routed_{j}',f'+ constrained routing: NMSE={metrics["candidate"]["nmse"][j]:.3f}')
            blank=fig.add_subplot(grid[4,0]);blank.axis('off')
            bg=float(np.mean(abs(parent[3].astype(np.complex128))**2)/mixture_power)
            blank.text(.04,.70,'Background target = 0',fontsize=11)
            blank.text(.04,.44,f'Estimated background / mixture\npower = {bg:.6g}\nRouting preserves this output.',fontsize=10,va='top')
            panel(grid[4,1],'parent_background','U-Net background',True)
            panel(grid[4,2],'routed_background','Routed background (unchanged)')
            bar=fig.colorbar(image,ax=axes,location='right',shrink=.8,pad=.015)
            bar.set_label('Mean FFT-bin power / whole-mixture power (dB)')
            for path_out in files[:3]:fig.savefig(path_out,dpi=300,facecolor='white')
            plt.close(fig)
        title='동일 전력 세 성분의 개선 사례' if index==7 else 'FPV가10dB 약한 조건의 악화 사례'
        files[4].write_text(f'''# {title}: B{index:02d}

원래 고정24개 개발 문맥 중 B7은 사전 지정 동일 전력 세 성분 사례, B4는 사후 분석에서
확인된 약신호 악화 사례다. 둘을 함께 제시하며 대표성·독립시험으로 주장하지 않는다.
전체20.8896ms를 잘라내지 않고 표시한다. 같은 VTSBW=10 원녹음 시간 분리 B 자료다.

첫 행은 혼합 I/Q다. 다음 세 행은 각 원기록 기여 파형(정답), 부모99 U-Net+C8/Hann,
그 출력을 제약 PSD 군집으로 재배정한 파형이다. 마지막 행은0인 배경 정답과 실제
배경 출력이다. 재배정은 배경을 바꾸지 않는다. 각 기체명은 평가용 정답 표지이며
분리기의 기종 식별 결과가 아니다. 원기록에는 수신 잡음도 포함된다.

평가용 정답 대응은 각 방법에 대해 전체 구간에서 한 번만 선택한 순열이다.
짧은 구간마다 정답으로 출력 순서를 바꾸거나 진폭·위상을 정답에 맞추지 않았다.
출력 재배정 알고리즘은 정답과 기체명을 사용하지 않는다.

공통 RF 좌표의0MHz는5780MHz다. 표본률100MS/s,1024점 FFT,
sqrt-Hann, hop1024(중첩0), 총2040프레임이다.
FFT 제곱을 FFT길이×창 에너지로 나눈 다음 전체 혼합 전력으로 나눈다.
표시할 때 인접2주파수 빈×2시간 프레임의 **선형 전력**을 평균하고10log10을 취한다.
표시 픽셀은195.3125kHz×20.48μs다. 같은 분석과 표시 연산을 모든 패널에 적용했다.
이미지는 nearest 표시이며 수치 보간·denoise·선택적 평활화는 없다.

모든 패널은 같은 색 범위 {lower:.3f}~{upper:.3f}dB를 사용한다.
이 범위는 B7/B4 두 그림 전체에 공통으로 고정했다. 범위 밖 빈 비율은
JSON에 보존했다. 패널별 자동 색상 정규화나 정답에 대한 개별 진폭 정규화는 없다.
색이 비슷해 보이는 것은 위상·I/Q 파형 일치의 증거가 아니므로 표시된 원파형 NMSE와
함께 해석한다. 이 STFT 표시 집계는 학습·평가 파형을 변경하지 않는다.

일반 연구 검토용 임시 그림이며 특정 학술지 규격 준수 판정은 아니다.
그림 작성 절차: Scientific Visualization 스킬. 도구 출처:
[Kassis et al., Scientific Agent Skills (2026)](https://doi.org/10.48550/arXiv.2609.00065).
''')
        manifest=dict(status='EXPORTED_PENDING_VISUAL_INSPECTION',case=index,title=title,
            selection='B7 fixed equal-power triple; B4 disclosed adverse weak-FPV case; both plotted',
            data_scope='Same-record B development, not independent confirmation',
            source_sha256=sha(Path(__file__)),input_sha256={str(p):sha(p) for p in (SOURCE/'RESULT.json',ROUTING/'RESULT.json',path,order_path)},
            full_samples=LENGTH,fs_hz=FS,n_fft=FFT,hop=FFT,window='sqrt periodic Hann',
            display_average_linear_power_bins=[2,2],display_pixel_hz=FS/FFT*2,display_pixel_ms=FFT/FS*2*1000,
            whole_mixture_power=mixture_power,color_range_db=[lower,upper],clipped_fraction=clipping,
            per_panel_normalization=False,reference_gain_phase_fitting=False,global_reference_label_mapping_only=True,
            parent_metrics=metrics['parent'],routed_metrics=metrics['candidate'],background_power_relative=bg,
            size_inches=[12,11],png_dpi=300,matplotlib_version=matplotlib.__version__,
            alt_text=f'{title}. 혼합과 세 정답, 부모와 출력 재배정 후 STFT를 동일 색 범위로 비교한다. 배경 출력도 포함한다.',
            exports={p.name:sha(p) for p in files if p.suffix!='.json'})
        files[3].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        print('FULL_STFT_EXPORTED',index,clipping,flush=True)


if __name__=='__main__':main()
