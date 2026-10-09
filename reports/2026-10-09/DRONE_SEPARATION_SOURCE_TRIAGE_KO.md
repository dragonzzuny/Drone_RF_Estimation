# 드론 분리 선행의 입력·출력 범위 재확인

확인일: 2026-10-09. 현재 GPU 학습과 별도로 문헌을 확인했다. 아래 검색 후보로 실행 중인 모델·자료·선택 규칙을 변경하지 않았다.

## 본문으로 확인한 사례

Ni와 Zhou의 2025년 Scientific Reports 논문, **Blind source separation and unmanned aerial vehicle classification using CNN with hybrid cross-channel and spatial attention module**는 개선 FastICA와 CNN 분류를 결합한다. 그러나 데이터는 **40 GHz 레이더의 RCS 시퀀스**다. 분리된 시퀀스를 GASF 이미지로 바꾸어 8종을 분류한다. 따라서 제목에 드론·분리가 있다는 이유로 수동 수신 통신 I/Q의 단일 채널 파형 복원과 같은 문제로 취급하지 않는다. 이 구분은 우월성 또는 선행 부재의 주장이 아니다. [공개 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12218027/), [출판사 DOI](https://doi.org/10.1038/s41598-025-07946-y).

## 원문 확인이 남은 후보

| 후보 | 이번에 확인한 근거 | 아직 확정하지 않는 내용 |
|---|---|---|
| Xue·Huang·Liu, Research on Blind Separation and Classification of UAV Frequency-Hopping Signals Under Hybrid Interference Environments, AIITA 2026 | 출판사 등록 Crossref 응답에서 제목·저자·DOI·2026-04-10 날짜 확인 | 관측 채널 수, 데이터의 실측/합성 범위, 파형 복원 지표, 비교 방법. IEEE 본문 접근 실패로 타 사이트 요약을 방법론 근거로 채택하지 않음 |
| CN122348790A | 검색 결과에 나타난 특허 문헌 식별자를 후속 확인 대상으로 기록 | 공식 원문·등록 정보와 공개 PDF 미확인. Google Patents 해당 주소는 HTTP404. 제3자 AI 요약의 구조·성능 수치를 검증된 근거로 인용하지 않음 |

첫 후보의 [DOI](https://doi.org/10.1109/AIITA69518.2026.11567196)와 [출판사 등록 메타데이터](https://api.crossref.org/works/10.1109/AIITA69518.2026.11567196)를 남긴다. 두 번째는 확정 선행 목록이나 논문 참고문헌에 아직 편입하지 않는다.

## 현재 연구에 적용할 판별 기준

새 방법을 비교 대상으로 편입하기 전에 실제 관측 채널 수, 복소 위상 사용 여부, 추론 시 단독 참조 파형·기종 정보를 요구하는지, 출력이 I/Q 파형인지 분류/분할 결과인지부터 확인한다. 합성 개수와 물리 드론 대수, 복원 점수와 식별 점수를 각각 구분한다. 현재 등록된 긴 복소문맥 비교는 이 문헌 검색과 관계없이 기존 규약대로 진행한다.
