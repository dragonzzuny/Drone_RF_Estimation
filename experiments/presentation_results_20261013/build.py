"""Rebuild the talk around the completed result, with condition-shift audits in backup.

No training, new inference, or raw I/Q access. Inputs are frozen presentation evidence.
"""
from pathlib import Path
import copy, csv, json, re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'reports/2026-10-13/presentation_results'
DEST.mkdir(parents=True, exist_ok=True)
NAME = '드론_RF_신호분리_성과중심_개정본'
OLD = json.loads((HERE/'inputs/original_records.json').read_text())
OLDINDEX = json.loads((HERE/'inputs/original_index.json').read_text())
OLDNOTES = re.split(r'\n## \d+장 — [^\n]+\n', (HERE/'inputs/original_notes.md').read_text())[1:]
ROWS = list(csv.DictReader((HERE/'inputs/PAPER_EVIDENCE_TABLE.csv').open()))
for font, file in [('Pretendard','Pretendard-Regular.ttf'), ('Pretendard-Bold','Pretendard-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(font, str(Path('/home/pyj/.fonts/pretendard')/file)))
INK='17313B'; TEAL='007C83'; BLUE='35659B'; GRAY='647986'; ORANGE='BA4A00'
WHITE='FFFFFF'; PALE='F1F5F6'; LT='E7F3F2'; LB='EAF0F7'; LO='FAEFE7'; GRID='DAE3E7'
slides=[]

def add(q): slides[-1]['records'].append(q)
def text(x,y,w,h,t,size=18,bold=False,color=INK,align=1):
    add(dict(kind='text',x=x,y=y,w=w,h=h,t=t,size=size,bold=bold,color=color,align=align))
def box(x,y,w,h,fill=PALE): add(dict(kind='box',x=x,y=y,w=w,h=h,fill=fill,line=None,round=True))
def line(x1,y1,x2,y2,color=GRAY,width=1.5,arrow=False,dash=False):
    add(dict(kind='line',x1=x1,y1=y1,x2=x2,y2=y2,color=color,width=width,arrow=arrow,dash=dash))
def start(section,title,sub='',notes='',appendix=False,dark=False):
    slides.append(dict(title=title,section=section,records=[],notes=notes,appendix=appendix,dark=dark))
    text(.65,.32,11.9,.3,section,13,False,'93D7D6' if dark else TEAL)
    text(.65,.91,12,.71,title,31,True,WHITE if dark else INK)
    if sub: text(.65,1.76,12,.44,sub,18,False,'C8DEDF' if dark else GRAY)
def body_note(t,y=6.51): text(.78,y,11.82,.45,t,14,False,GRAY)
def node(x,y,w,h,title,body,color=TEAL):
    box(x,y,w,h,LT if color==TEAL else LB)
    text(x+.18,y+.21,w-.36,.43,title,22,True,color)
    text(x+.18,y+.88,w-.36,h-.95,body,19)
def clone(i,title,sub,section=None):
    old=copy.deepcopy(OLD[i])
    start(section or OLDINDEX[i]['section'],title,sub,OLDNOTES[i].split('\n## 発')[0])
    for q in old:
        if q['kind']=='text' and (q['y']<2.15 or q['y']>=6.98): continue
        if q['kind']=='image': q['filename']=str(HERE/'assets/actual_stft.png')
        add(q)
def replace(old,new):
    found=0
    for q in slides[-1]['records']:
        if q['kind']=='text' and q['t']==old:q['t']=new;found+=1
    assert found, old
def value(group,method,count,key='mean_nmse',parent='B99'):
    return float(next(r for r in ROWS if (r['group'],r['parent'],r['method'],int(r['count']))==(group,parent,method,count))[key])
def table(headers,rows,widths,x=.76,y=2.7,rowh=.70,size=18,highlights=()):
    x0=x; box(x,y,sum(widths),.52,INK)
    for t,w in zip(headers,widths):text(x+.13,y+.13,w-.26,.30,t,15,True,WHITE);x+=w
    for i,row in enumerate(rows):
        x=x0;yy=y+.52+i*rowh;box(x,yy,sum(widths),rowh,LT if i in highlights else PALE if i%2==0 else WHITE)
        for t,w in zip(row,widths):text(x+.13,yy+.16,w-.26,rowh-.18,str(t),size,i in highlights);x+=w

# Main talk: 14 slides. The count-method slide contains the method, not a shifted-domain scoreboard.
start('연구 발표','드론 RF 혼합신호의 분리 및 복원',
      '복소 U-Net과 신호 연결 모델을 결합한 파형 복원',dark=True,
      notes='연구 질문은 여러 신호가 합쳐진 수신 파형에서 각 성분의 I/Q를 연속적으로 복원할 수 있는가입니다. 공개 RFUAV의 개별 수신 기록을 합성한 지도학습 연구입니다. 핵심 성과는 같은 녹음의 미학습 시간 구간에서, 고정 주파수 분포 연결에 비해 학습 연결이 긴 구간의 평균 복원 오차를 줄인 것입니다. 실제 동시 수신·미학습 기종 일반화를 입증했다는 뜻은 아닙니다.')
text(.78,2.80,11.7,1.18,'짧은 구간에서 파형을 복원하고\n같은 신호를 시간에 따라 연결한다',33,True,WHITE)
for x,title,body in [(.78,'혼합 I·Q','여러 신호가 겹친 입력'),(4.96,'성분별 복원','U-Net으로 파형 추정'),(9.14,'시간 연결','같은 신호 조각 정렬')]:
    box(x,4.73,3.41,1.35,'244650');text(x+.19,4.94,3.03,.40,title,23,True,WHITE);text(x+.19,5.54,3.03,.30,body,16,False,'C8DEDF')
for x in [4.28,8.46]:line(x,5.36,x+.53,5.36,'93D7D6',1.7,True)

clone(1,'혼합 신호와 정답 파형 구성','RFUAV 5기종 · 같은 RF 대역 안에서 합성 · 원래 중심주파수 간격 유지','데이터 구성')
body_note('두 신호: 4가지 기종 조합  |  세 신호: 1가지 기종 조합  |  표본률 100 MS/s')
clone(2,'혼합 I·Q에서 신호별 파형으로','시간·주파수 변환 → 성분 추정 → 역변환','방법 ① 파형 복원')
replace('출력: 신호 3개 + 남은 부분\n긴 구간의 평균 전력도 참고\n위상을 바꾼 8개 복원 결과를 평균','신호 3개 + 배경 출력\n긴 구간의 전력 특징을 함께 사용\n8개 위상의 복원 결과를 정렬·평균')
clone(3,'U-Net을 이용한 복소 성분 추정','넓은 패턴과 세부 정보를 함께 읽어 신호별 I·Q 지도 생성','분리 모델')
replace('① 지도를 줄이며 전체 패턴 파악','① 축소 경로: 넓은 패턴 추출')
replace('② 확대하며 세부 정보 복원','② 확대 경로: 세부 값 복원')
replace('출력: 신호별 I·Q 지도','출력: 신호별 실수부·허수부 지도 → iSTFT로 파형 복원')
slides[-1]['references']=['https://arxiv.org/abs/1505.04597']

start('학습 방법','정답 파형으로 분리와 개수 추정을 함께 학습',
      '합성 전 신호를 정답으로 사용 · 추론할 때는 혼합 신호만 입력',
      notes=OLDNOTES[4]+'\n파형 손실은 성분별 NMSE와 coherence, 비활성 출력 및 배경 항을 포함합니다. 개수 CE의 가중치는 0.1입니다. 긴 문맥의 평균 전력 특징은 원래 시간 순서를 보존하지 않습니다. 뒤의 신호 연결 모델이 복원 조각의 관계를 따로 학습합니다.')
node(.78,2.63,5.67,2.34,'파형 복원 학습','출력과 정답의 대응을 선택\n파형 오차와 유사도를 함께 평가\n신호가 없는 출력은 0에 가깝게',TEAL)
node(6.86,2.63,5.67,2.34,'긴 구간의 보조 정보','약 20.9 ms의 혼합에서\n주파수별 전력과 전체 세기를 요약\n분리 모델과 개수 분류기에 전달',BLUE)
text(.91,5.39,11.55,.55,'파형 복원 손실 + 0.1 × 신호 개수 분류 손실',25,True,TEAL)
body_note('혼합은 긴 구간에서 전력을 맞춘 뒤 자르며, 짧은 창마다 다시 정규화하지 않음')
slides[-1]['references']=['https://arxiv.org/abs/1703.06284']

start('방법 · 신호 개수','혼합 신호에서 성분 수 추정',
      '전력 특징으로 1개·2개·3개를 분류하고, 연결할 신호 그룹 수 결정',
      notes='약 20.9 ms 혼합에서 주파수별 상대 전력 64개와 상대 세기 1개를 계산하고 log 변환 및 시간 평균을 적용합니다. 평균 특징을 반복한 입력은 1차원 합성곱 특징 신경망을 거쳐 128차원으로 요약됩니다. Linear128→128→3에서 최대 점수를 선택합니다. 합성 원기록 성분 수가 정답이며 물리 드론 대수를 검증한 결과가 아닙니다. 파형 출력은 3개와 배경 1개로 고정되고, 예측 개수가 출력을 지우지 않습니다. 이후 연결에서 그룹 수로 사용합니다. 조건별 정확도는 부록 B에 모두 보존했습니다.')
items=[('혼합 I·Q','약 20.9 ms'),('전력 특징','주파수 64개 + 세기 1개'),('특징 신경망','시간 평균 → 1D 합성곱'),('개수 분류','1·2·3개 중 선택')]
for i,(a,b) in enumerate(items):
    x=.76+i*3.12;box(x,2.65,2.74,1.63,LT if i%2==0 else LB)
    text(x+.13,2.89,2.48,.39,a,22,True,TEAL if i%2==0 else BLUE,2)
    text(x+.13,3.62,2.48,.38,b,15.5,False,INK,2)
    if i<3:line(x+2.82,3.47,x+3.04,3.47,TEAL,1.6,True)
node(.80,4.89,5.64,1.39,'학습 정답','혼합에 사용한 원신호 수',TEAL)
node(6.86,4.89,5.64,1.39,'추론 결과의 역할','복원 조각을 묶을 그룹 수',BLUE)
body_note('신호용 파형 출력은 항상 3개이며, 개수 분류는 1~3개 합성 성분을 대상으로 함')

clone(6,'Transformer로 같은 신호 조각 연결','파형 값은 유지하고, 시간에 따른 출력 배정을 학습','방법 ② 신호 연결')
replace('관계 비교 신경망','조각 간 관계 학습')
body_note('4개 Transformer 층 · 약 88만 학습 변수 · U-Net 고정 후 연결 모델만 학습')

start('평가 설계','조건을 맞추고 연결 방법의 효과 비교',
      '같은 녹음의 미학습 시간 구간 · 동일한 분리 모델과 혼합 조건 사용',
      notes='본문은 10 MHz 전송 대역폭 기록의 시간 분리 개발 평가입니다. 앞 구간으로 학습하고 필터 지원까지 고려해 1초 이상의 간격을 두고 뒤 구간을 평가했습니다. 독립 원녹음 평가는 아닙니다. U-Net은 100 epoch 완료 후 630개 짧은 개발 혼합의 오차로 epoch99를 선택했습니다. 연결기는 별도 FIT48 출력에서 50 epoch 학습했습니다. 여기의 긴 구간 평가 24조건은 단일5/두12/세7개이며 독립 녹음 24개를 뜻하지 않습니다. 전부 한 seed의 반복 확인된 개발 결과입니다. 녹음과 전송 대역폭이 함께 바뀌는 확장 결과는 부록 A에 있습니다.')
text(.86,2.50,11.59,.39,'원녹음의 시간 구간을 먼저 나누고, 각 구간 안에서 혼합 생성',22,True)
box(.87,3.14,4.67,.65,LT);text(1.07,3.32,4.27,.33,'앞 구간 · 학습',22,True,TEAL,2)
line(5.73,3.46,7.22,3.46,GRAY,1.8);text(5.67,3.74,1.67,.40,'1초 이상 간격',14,False,GRAY,2)
box(7.44,3.14,4.99,.65,LB);text(7.63,3.32,4.61,.33,'뒤 구간 · 개발 평가',22,True,BLUE,2)
table(['구성','학습 및 평가'],[
    ['파형 분리 모델','100 epoch 학습 → 검증으로 epoch 99 선택'],
    ['신호 연결 모델','분리 모델을 고정하고 50 epoch 학습'],
    ['전체 구간 평가','약 20.9 ms · 두 신호 12조건 / 세 신호 7조건']], [3.12,8.69],y=4.26,rowh=.53,size=17)
body_note('핵심 지표: 복원 오차 NMSE ↓ · 복소 SI-SDR ↑ · 최약 신호의 NMSE ↓')

start('결과 ① 전체 복원 성능','학습 연결로 긴 구간의 복원 오차 감소',
      '같은 U-Net 출력에서 연결 방법만 변경 · 전체 평가 조건의 평균 NMSE',
      notes='이 그림은 같은 녹음의 미학습 시간 구간에서 학습 연결의 효과를 비교합니다. 두 신호12조건과 세 신호7조건을 모두 포함한 평균입니다. 주파수 분포를 비교하는 고정 연결 대비 각각37.6%,50.6%의 상대 NMSE 감소입니다. 정확도가37.6% 또는50.6%라는 뜻이 아닙니다. 두 신호12조건 중4조건은 고정 연결보다 악화하여 부록 B에 표시했습니다. 단일5조건은 연결 전후 값이 같습니다. 통계적 유의성이나 독립 녹음 일반화를 주장하지 않습니다.')
add(dict(kind='chart',x=.77,y=2.56,w=7.99,h=3.83,
    labels=['두 신호','세 신호'],series=[dict(name=n,values=[value('B24',m,k) for k in (2,3)]) for m,n in [('parent','그대로 연결'),('psd','주파수 분포 연결'),('learned50','학습 연결')]],
    colors=['94A5AE',BLUE,TEAL],ymax=1.6,unit=.4))
box(9.02,2.70,3.51,3.54,LT)
text(9.24,2.96,3.08,.62,'주파수 분포 연결 대비',18,True,TEAL)
text(9.24,3.91,3.08,.45,'37.6% 감소',29,True,TEAL)
text(9.24,4.46,3.08,.30,'두 신호 평균 NMSE',17)
text(9.24,5.14,3.08,.45,'50.6% 감소',29,True,TEAL)
text(9.24,5.70,3.08,.30,'세 신호 평균 NMSE',17)
body_note('동일 녹음의 미학습 시간 구간 · 두 신호 12조건 / 세 신호 7조건 · NMSE는 낮을수록 좋음')

start('결과 ② 보완 지표','파형 유사도와 약한 신호의 평균 오차도 개선',
      '같은 평가 조건에서 주파수 분포 연결과 학습 연결 비교',
      notes='복소 SI-SDR은 전체 파형의 투영에 기반한 스케일 불변 지표이고, NMSE는 실제 크기 오차를 반영하므로 함께 평가합니다. 표의 각 행은 동일한 두 신호12조건 또는 세 신호7조건의 평균입니다. 최약 신호는 각 혼합에서 정답 전력이 가장 낮은 성분이며, 동일 전력에서는 구현의 선택 규칙이 적용됩니다. 평균 개선이며 모든 조건이나 모든 약한 패킷이 개선했다는 뜻은 아닙니다.')
table(['평가 지표','신호 수','주파수 분포 연결','학습 연결'],[
    ['복소 SI-SDR ↑','2개',f'{value("B24","psd",2,"mean_si_sdr"):.2f} dB',f'{value("B24","learned50",2,"mean_si_sdr"):.2f} dB'],
    ['복소 SI-SDR ↑','3개',f'{value("B24","psd",3,"mean_si_sdr"):.2f} dB',f'{value("B24","learned50",3,"mean_si_sdr"):.2f} dB'],
    ['최약 신호 NMSE ↓','2개',f'{value("B24","psd",2,"weakest_nmse"):.3f}',f'{value("B24","learned50",2,"weakest_nmse"):.3f}'],
    ['최약 신호 NMSE ↓','3개',f'{value("B24","psd",3,"weakest_nmse"):.3f}',f'{value("B24","learned50",3,"weakest_nmse"):.3f}']],
    [3.30,1.50,3.50,3.52],y=2.69,rowh=.70,size=21,highlights=(1,3))
body_note('동일 녹음의 미학습 시간 구간 · 전체 조건 평균 · 크기 오차와 파형 유사도를 함께 확인')

start('결과 ③ 시간에 따른 복원','같은 신호의 송신 구간을 연속적으로 복원',
      '세 신호의 동일 전력 혼합 예시 · 약 20.9 ms 전체 구간',
      notes='Avata2, FPV COMBO, Mavic3 Pro를 같은 전력으로 혼합한 예시입니다. 검정은 정답, 회색 점선은 그대로 연결, 청록 점선은 학습 연결입니다. 전력은 각 정답의 평균 전력으로 나누어 표시했습니다. 전력 곡선은 시간별 활동을 보여 주며 I/Q 위상 복원의 증거는 다음 장 파형과 정량 지표에서 확인합니다. 세 신호의 전체 구간 학습 연결 NMSE는 .227270,.137408,.102321입니다. 잘 복원된 사례로 선택했고 모집단 대표 무작위 표본은 아닙니다.')
add(dict(kind='image',x=.71,y=2.43,w=11.93,h=4.05,filename=str(HERE/'assets/full_power_B07.png')))
body_note('검정: 정답 · 회색: 그대로 연결 · 청록: 학습 연결  |  신호별 정답 평균 전력으로 정규화')

clone(9,'두 신호의 I·Q 복원 예시','Avata2 + Mavic3 Pro · Avata2의 전력은 Mavic3 Pro의 1/10','결과 ④ 실제 파형')
slides[-1]['notes']+='\n그림은 선택한 사례의 일부 약5.12 μs 확대입니다. 신호마다 확대 위치가 다릅니다. 약20.9 ms 전체 구간 NMSE는 Avata2 .107142, Mavic3 Pro .010964입니다. 정답과 복원을 같은 정답 RMS로 표시했으며 정답을 이용한 배율·위상 맞춤이나 평활화를 하지 않았습니다. 이 예시는 두 신호의 복원 자체를 보여 주며 고정 PSD 연결과 학습 연결의 차이를 보여 주는 사례는 아닙니다.'
body_note('선택한 복원 사례의 일부 확대 · 전체 구간 NMSE: Avata2 0.107 / Mavic3 Pro 0.011',y=6.68)
clone(10,'세 신호의 I·Q 복원 예시','Avata2 + FPV COMBO + Mavic3 Pro · 세 성분을 같은 전력으로 혼합','결과 ④ 실제 파형')
slides[-1]['notes']+='\n각 행은 선택한 사례의 일부 약5.12 μs 확대이며 신호별 확대 위치가 다릅니다. 약20.9 ms 전체 구간 NMSE는 .227270,.137408,.102321입니다. 추론에서 정답을 이용한 배율·위상 보정은 없습니다.'
body_note('선택한 복원 사례의 일부 확대 · 전체 NMSE: Avata2 0.227 / FPV 0.137 / Mavic3 Pro 0.102',y=6.68)

start('결론','파형 복원과 신호 연결의 결합 효과 확인',
      'RFUAV 합성 혼합 · 같은 녹음의 미학습 시간 구간에서 검증',dark=True,
      notes='발표의 기여는 복소 U-Net으로 성분 파형을 추정하고, 별도 연결 모델로 같은 신호의 시간적 배정을 개선한 두 단계 구성과 그 통제 비교입니다. 2/3성분 긴 구간 평균 NMSE는0.1430/0.2421이며, 고정 주파수 분포 연결 대비37.6%/50.6% 감소했습니다. 기존 U-Net이나 Transformer 자체의 발명을 주장하지 않습니다. 한 seed의 반복 확인된 개발 결과입니다. 조건 변화 결과와 범위는 부록에 보존했으며, 같은 조건의 독립 녹음·미학습 기종·실제 동시 수신에 대한 성능은 별도 검증이 필요합니다.')
for x,title,body in [(.79,'파형 복원','복소 U-Net으로\n성분별 I·Q 추정'),(4.97,'신호 연결','Transformer로\n같은 신호 조각 배정'),(9.15,'정량 검증','파형 오차 · 유사도\n약한 신호 오차 평가')]:
    box(x,2.68,3.38,1.95,'244650');text(x+.18,2.94,3.02,.43,title,24,True,WHITE);text(x+.18,3.65,3.02,.67,body,20,False,'C8DEDF')
text(.86,5.05,11.63,.64,'평균 복원 오차   두 신호 37.6% ↓   세 신호 50.6% ↓',28,True,'93D7D6')
text(.86,6.00,11.63,.63,'주파수 분포 연결 대비 상대 감소율\n확장 검증: 독립 녹음 · 미학습 기종 · 실제 동시 수신',17,False,'C8DEDF')

# Backup results remain visible and accurately scoped; no counterevidence is deleted.
start('부록 A · 조건 변화','녹음·전송 대역폭 변화에 대한 추가 평가',
      '평균 복원 오차(NMSE) ↓ · 기존 5기종 유지 · 각 패널 안에서 비교',
      notes=OLDNOTES[11],appendix=True)
text(.83,2.44,5.62,.40,'① 같은 모델의 조건 변화 평가',21,True)
text(.83,3.00,5.62,.44,'동일 U-Net·연결 모델 · 10 → 20 MHz',16,False,GRAY)
text(6.90,2.44,5.61,.40,'② 별도 모델의 연결 방법 비교',21,True)
text(6.90,3.00,5.61,.44,'별도 U-Net 가중치 · 조건 변화 자료',16,False,GRAY)
add(dict(kind='chart',x=.78,y=3.54,w=5.68,h=2.87,
    labels=['두 신호','세 신호'],series=[dict(name=n,values=[value(g,'learned50',k) for k in (2,3)]) for g,n in [('B24','학습 조건 내'),('C24','기록·대역폭 변화')]],
    colors=[TEAL,BLUE],ymax=.85,unit=.2))
add(dict(kind='chart',x=6.84,y=3.54,w=5.68,h=2.87,
    labels=['두 신호','세 신호'],series=[dict(name=n,values=[value('C24',m,k,parent='native_e2') for k in (2,3)]) for m,n in [('psd','주파수 분포 연결'),('matched50','맞춤 연결 학습')]],
    colors=[BLUE,ORANGE],ymax=.85,unit=.2))
body_note('녹음과 대역폭의 영향은 분리되지 않음 · 같은 조건의 별도 원녹음 평가와 구분')
slides[-1]['notes']+='\n중요한 교정: 이 결과는 동일 수신 조건에서 원녹음만 바꾼 실험이 아닙니다. VTSBW는 영상 전송 신호의 명목 대역폭이며 수신기 표본률은100MS/s를 유지했습니다. 기존5기종에는 같은 기종×대역폭에서 두 번째 원녹음 묶음이 없어 그 독립 평가를 구성하지 못했습니다.'

start('부록 B · 상세 평가','개수 추정과 조건별 결과',
      '개수는 합성 원신호 수 · 파형 성능은 모든 평가 조건의 평균',appendix=True,
      notes='개수 표는 B99+learned50의 동일한 고정 모델로 계산한 B24/C24입니다. 학습 조건 내 시간 분리에서는5/5,12/12,7/7이며, 원녹음과 전송 대역폭이 함께 바뀐 자료에서는3/5,4/12,5/7입니다. 총24/24와12/24이고, 물리 드론 대수를 검증한 지표가 아닙니다. 전반 평균 개선과 개별 악화는 동시에 성립합니다. 동일 조건 두 신호12건 중5건 개선·4건 악화·3건 동률, 세 신호7건 중6건 개선·1건 동률입니다. 1/2/3개 합성 평가일 뿐 0개·4개 이상·미학습 기종의 성능은 포함하지 않습니다.')
table(['평가 조건','1개','2개','3개','전체'],[
    ['같은 녹음의 미학습 시간','5/5','12/12','7/7','24/24'],
    ['다른 녹음 + 다른 전송 대역폭','3/5','4/12','5/7','12/24']],
    [5.30,1.51,1.51,1.51,1.99],y=2.69,rowh=.73,size=18)
node(.79,5.02,5.62,1.40,'본문의 평균 집계','두 신호 12조건 · 세 신호 7조건 전체',TEAL)
node(6.87,5.02,5.62,1.40,'두 신호의 개별 결과','12조건 중 4조건은 고정 연결보다 악화',BLUE)

clone(12,'복원 파형의 패킷 판독 확인','DJI FPV 패킷 1건 · 전력을 바꾼 4조건 · 연결 모델 적용 전 검사','부록 C · 활용 가능성')
slides[-1]['appendix']=True
replace('확인한 범위','판독 결과')
body_note('혼합에서도 같은 패킷이 판독되므로, 이 검사만으로 탐지 성능 향상을 주장하지 않음')
slides[-1]['references']=['https://github.com/RUB-SysSec/DroneSecurity']

assert len(slides)==17 and sum(not s['appendix'] for s in slides)==14
overflow=[]
def wrap(t,font,size,width):
    result=[]
    for para in t.split('\n'):
        current=''
        for word in para.split(' '):
            trial=current+(' ' if current else '')+word
            if pdfmetrics.stringWidth(trial,font,size)>width and current:result.append(current);current=word
            else:current=trial
        result.append(current)
    return result
for i,s in enumerate(slides,1):
    s['page']=i
    s['notes']=s['notes'].split('\n## 발표에 남긴 주요 용어')[0].strip()
    for q in s['records']:
        if q['kind']=='text':
            font='Pretendard-Bold' if q['bold'] else 'Pretendard'
            q['lines']=wrap(q['t'],font,q['size'],q['w']*72)
            need=q['size']*(1+1.13*(len(q['lines'])-1))
            if need>q['h']*72+4:overflow.append(dict(page=i,text=q['t'],needed=need,available=q['h']*72))
    s['records'].append(dict(kind='text',x=12.03,y=7.01,w=.62,h=.30,t=f'{i:02}',lines=[f'{i:02}'],size=12,bold=False,color='C8DEDF' if s['dark'] else GRAY,align=3))
    if s.get('references'):
        s['notes']+='\n\n방법의 선행 근거:\n'+'\n'.join(s['references'])
    for q in s['records']:
        if q['kind']=='image':q['filename']=str(Path(q['filename']).relative_to(HERE))

(HERE/'DECK.json').write_text(json.dumps(dict(name=NAME,output=str(DEST),main_slides=14,slides=slides),ensure_ascii=False))
(HERE/'qa/TEXT_OVERFLOW.json').write_text(json.dumps(overflow,ensure_ascii=False,indent=2))
notes=['# 드론 RF 신호 분리 발표 설명','본문 14장 / 추가 검증 부록 3장. 화면은 핵심 결과를 중심으로 구성하고, 상세 조건과 해석은 발표 노트에 보존했습니다.']
for s in slides: notes += [f'\n## {s["page"]}장 — {s["title"]}',s['notes']]
notes += ['\n## 참고문헌','- Ronneberger, Fischer, Brox (2015). U-Net. https://arxiv.org/abs/1505.04597','- Kolbæk, Yu, Tan, Jensen (2017). Utterance-level permutation invariant training. https://arxiv.org/abs/1703.06284','- RUB-SysSec. DroneSecurity (NDSS 2023). https://github.com/RUB-SysSec/DroneSecurity','선행 문헌의 개념과 공개 수신기를 참고했으며, 본문 성능 수치는 프로젝트의 완료된 실험에서 가져왔습니다.']
(DEST/'발표자_설명.md').write_text('\n\n'.join(notes))
print(json.dumps(dict(slides=len(slides),main=14,appendix=3,overflow=overflow),ensure_ascii=False))
