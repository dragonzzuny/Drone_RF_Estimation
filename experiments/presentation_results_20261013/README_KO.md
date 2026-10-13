# 성과 중심 발표 개정

사용자 요청에 따라 기존 핵심 개정본을 발표 흐름과 편집 가능한 PowerPoint로 재구성했다.
본문은 14장, 추가 검증 부록은 3장이다.

- 6장은 개수 추정의 원리와 연결에서의 역할만 설명한다.
- 동일 녹음의 미학습 시간 구간에서 얻은 성과를 본문의 평가 범위로 명시한다.
- 전체 조건의 평균 NMSE 감소, 복소 SI-SDR, 최약 신호 NMSE를 함께 제시한다.
- 선택한 실제 파형에는 확대 예시 표기와 전체 구간 NMSE를 붙였다.
- 녹음·전송 대역폭 동시 변화 결과, 조건별 개수 정확도, 악화 조건, 패킷 판독 결과는 부록에 보존했다.
- 원시 I/Q나 보류 기종을 읽거나 새 학습·추론을 수행하지 않았다. 진행 중 WaveNet은 변경하지 않았다.

산출물은 `reports/2026-10-13/presentation_results/`에 있다.
PPTX는 글자·구조도·막대그래프를 편집할 수 있고, 34개 파형 곡선도 PowerPoint의 기본 벡터 도형이다.
기존 SVG 삽입의 LibreOffice 호환 문제를 발견하여 원래 좌표를 DrawingML 경로로 직접 기록했다.
PDF는 최종 PPTX에서 내보냈으며 모든 글꼴이 포함됐다. 전 17장을 렌더링해 확인했다.

## 재생성

저장소 루트에서 실행한다. Python ReportLab/lxml, PptxGenJS, LibreOffice, Poppler, Pretendard가 필요하다.
`render.cjs`는 설치된 PptxGenJS를 우선 사용하며 로컬 설치 경로 또는 `DRONE_PPTXGEN_MODULE`을 지원한다.

```bash
python experiments/presentation_results_20261013/build.py
node experiments/presentation_results_20261013/render.cjs
python experiments/presentation_results_20261013/native_paths.py
```

생성된 PPTX를 LibreOffice에서 PDF로 내보낸 후 `verify.py`를 실행한다.
PowerPoint 패키지도 pptx 스킬의 `scripts/office/validate.py`로 검사했다.
큰 미리보기 이미지는 Git에 포함하지 않는다. 원문 출처와 수집 경로는 입력 해시 명세에 보존한다.
참고 문헌은 발표자 설명과 PPTX 노트에 기록했고, 화면에는 내부 코드·파일 경로를 노출하지 않는다.

검토 방법: scientific-slides, pptx, pdf 스킬의 구성·렌더링·구조 검증 절차.
