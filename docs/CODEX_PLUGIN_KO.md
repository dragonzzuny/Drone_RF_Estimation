# Claude Code용 Codex 플러그인

[OpenAI 공식 codex-plugin-cc](https://github.com/openai/codex-plugin-cc)를 사용자 범위로 설치했다. 버전은 1.0.6이며, 활성화·manifest 검사·Codex app-server 인증 검사를 통과했다. Node 22.22.1, Claude Code 2.1.281, Codex CLI 0.160.1 환경이다.

Claude Code는 현재 로그아웃 상태다. Claude 터미널에서 사용하려면 먼저 로그인한다.

```text
/login
/reload-plugins
/codex:setup
/codex:adversarial-review --background 실험의 변인통제, 데이터 누출, 수식과 코드 일치 여부를 검토해줘
/codex:status
/codex:result
```

설치된 플러그인의 Codex 실행 경로를 직접 사용한 읽기 전용 검토 3건을 완료했다. 두 번째는 세 신호 분리를 위한 정보 흐름과 수식의 구조 검토다. 새 혼합 실험에서 정상적인 유한 지표 경로의 학습 차단 오류는 찾지 못했고, 비유한 지표 처리·선택 지표 검산·기록별 보고 등을 지적했다. Claude 모델의 검토나 Claude UI를 거친 실행 완료로 표시하지 않는다.

검토에는 소스·실험 명세·기존 JSON만 허용했고, I/Q·체크포인트·GPU 접근과 실험 실행을 금지했다. 자동 종료 검토 게이트는 기존 비활성 상태로 유지했다.

[설치 상태](../reports/2026-10-10/CODEX_PLUGIN_INSTALL.json) · [검토 원문](../reports/2026-10-10/CODEX_PLUGIN_REVIEW.md) · [후속 검산](../reports/2026-10-10/CODEX_PLUGIN_REVIEW_FOLLOWUP.json)

[구조 검토 원문](../reports/2026-10-10/CODEX_ARCHITECTURE_REVIEW_KO.md) · [완료 결과의 추가 검산](../reports/2026-10-10/CODEX_PLUGIN_REVIEW_FOLLOWUP_KO.md)

세 번째 검토는 SepTDA 참고 RF 후보의 축·미분·학습·감사 코드다. 지적된 중첩 재계산 검사 공백을 실제 GPU 출력·gradient 비교로 보강했다. [검토와 반영](../reports/2026-10-10/SEPTDA_RF_REVIEW_RESPONSE_KO.md).
