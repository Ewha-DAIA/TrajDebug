# FALAT 재현 결과 (GPT-5.4-Mini, Who&When)

> 대상 논문: Rafi et al., **FALAT: Tracing Failures in LLM Agent Trajectories via Dependency-Guided Search**, arXiv:2606.00765
> 구현: `falat/` (저자 공개 코드 없음 → 본문 §2와 부록 A 프롬프트로 직접 구현). 구현 방법과 재현 가정 17개는 [`README.md`](README.md)에 정리했다.
> 작성일: 2026-10-02

## 1. 요약

- **Step 정확도는 논문 수치를 재현했다.** Hand-crafted 20.7–22.4 (논문 22.2), Algorithm-generated 44.4–45.2 (논문 42.0). 허용 오차(±1~±5) 곡선도 논문 Table 3과 비슷하다.
- **Agent 정확도는 논문보다 낮다.** Algorithm-generated는 61.9–63.5 (논문 66.5)로 약 3–5%p 차이지만, **Hand-crafted는 46.6–51.7 (논문 65.5)로 14–19%p 차이**가 남는다.
- Hand-crafted agent 격차의 주원인은 **step을 틀렸을 때 agent까지 같이 틀리는 구조**다. 틀린 예측이 정답보다 늦은 Orchestrator 단계(종료 메시지 포함)에 몰리고, HC 메시지의 76%가 Orchestrator라서 agent도 같이 틀린다. agent를 계산하는 방식을 바꿔도(LLM이 답한 agent 사용 등) 격차는 줄지 않았다.
- reasoning(`medium`)을 켜면 HC agent가 +5.2%p 올랐지만 격차를 메우지는 못했고, 출력 토큰은 약 2배가 됐다.

## 2. 실험 설정

| 항목 | 값 |
|---|---|
| 데이터 | Who&When — Hand-crafted 58개, Algorithm-generated 126개 (`data/whoandwhen/`, TrajDebug unified 형식. 원본과 정답 step 일치 확인) |
| 백본 | `openai/gpt-5.4-mini` (OpenRouter 경유) |
| 생성 설정 | temperature 미지정(모델 기본값). reasoning: 기본값(추론 토큰 0) / `medium` 두 조건 |
| FALAT 설정 | window k = 10, \|C_H\| = K = 3, clip 없음, API 재시도 3회, 파싱 재시도 3회 |
| 최종 답 | Stage 4 결과 t̃ / ã (Stage 3 결과 t̂ / α̂도 함께 보고, §5.2 참고) |
| Agent 판정 | 예측 step을 말한 agent(괄호 접미사 제거) vs `metadata.extra.mistake_agent`. 대소문자·밑줄 무시 |

## 3. 결과

### 3.1 논문 Table 2와 비교

| 데이터 | 조건 | Agent | Step |
|---|---|---|---|
| **Hand-crafted** (n=58) | 재현 · reasoning 기본값 | 46.6 | 20.7 |
| | 재현 · reasoning medium | 51.7 | 22.4 |
| | *논문 FALAT (GPT-5.4-Mini)* | *65.5* | *22.2* |
| **Algorithm-generated** (n=126) | 재현 · reasoning 기본값 | 61.9 | 44.4 |
| | 재현 · reasoning medium | 63.5 | 45.2 |
| | *논문 FALAT (GPT-5.4-Mini)* | *66.5* | *42.0* |

참고(논문 Table 2, 같은 백본): CHIEF HC 55.2 / 13.8, AG 59.0 / 27.0.

### 3.2 허용 오차 step 정확도 (논문 Table 3, Hand-crafted)

| | Exact | ±1 | ±2 | ±3 | ±4 | ±5 |
|---|---|---|---|---|---|---|
| 재현 · 기본값 | 20.7 | 27.6 | 27.6 | 34.5 | 44.8 | 46.6 |
| 재현 · medium | 22.4 | 29.3 | 29.3 | 32.8 | 43.1 | 44.8 |
| *논문 (GPT-5.4-Mini)* | *22.2* | *22.4* | *25.9* | *32.8* | *41.4* | *43.1* |

Algorithm-generated (논문은 보고하지 않음): 기본값 ±1 57.9, ±3 80.2, ±5 89.7 / medium ±1 52.4, ±3 77.0, ±5 88.9.

### 3.3 단계별 진단 (정답 step이 어디까지 살아남는가)

| | 정답 ∈ C (Stage 1) | 정답 ∈ C_H (Stage 3a) | t̂ 정확도 (Stage 3b) | t̃ 정확도 (Stage 4) | Stage 4: 고침 / 망침 |
|---|---|---|---|---|---|
| HC 기본값 | 82.8 | 39.7 | 20.7 | 20.7 | 0 / 0 |
| HC medium | 84.5 | 34.5 | 22.4 | 22.4 | 0 / 0 |
| AG 기본값 | 92.1 | 67.5 | 45.2 | 44.4 | 1 / 2 |
| AG medium | 93.7 | 73.8 | 41.3 | 45.2 | 7 / 2 |

- Stage 1은 정답을 대부분 후보에 포함시킨다(83–94%).
- **가장 큰 손실은 Stage 3a(C → C_H)**다. 특히 HC는 절반 이상이 여기서 빠진다.
- Stage 4는 HC에서 정답 여부를 한 번도 바꾸지 않았다. AG에서는 reasoning을 켰을 때만 순효과(+5)가 있었다.

## 4. Hand-crafted agent 격차 분석

| 지표 | 기본값 | medium |
|---|---|---|
| step 정답 → agent 정답 | 10 / 12 | 12 / 13 |
| **step 오답 → agent 정답** | **17 / 46 (37%)** | **18 / 45 (40%)** |
| step 오답 중 정답보다 늦은 예측 | 33 / 46 | 34 / 45 |
| 마지막 step(종료 메시지)을 예측 | 12 | 13 |
| 최다 오류: 정답 WebSurfer → 예측 Orchestrator | 14 | 13 |

1. **agent가 step에 묶여 있다.** 논문은 α̂를 "t̂에 해당하는 agent"로 정의하고(§2.3, §2.4), 부록 프롬프트에도 agent를 따로 판단하는 단계가 없다. 그래서 step을 틀리면 agent는 우연히만 맞는다. 논문 수치(step 22.2, agent 65.5)가 나오려면 step 오답의 약 55%에서 agent가 맞아야 하는데, 재현은 37–40%다.
2. **틀린 step이 Orchestrator에 몰린다.** HC 메시지 2,993개 중 76.3%가 Orchestrator, 19.0%가 WebSurfer다. 그런데 정답 agent는 WebSurfer가 57%(33/58)로 가장 많다. 참고로 항상 "WebSurfer"라고 답하면 53.4%로, 재현 결과(46.6–51.7)보다 높다.
3. **원인보다 늦은 증상을 고른다.** step 오답의 약 3/4이 정답보다 늦은 step이고, 전체 예측의 약 1/5(12–13/58)이 궤적의 마지막 step(주로 실행 종료 메시지 `Orchestrator (termination condition)`)이다. FALAT가 구분하려는 "source vs downstream symptom"이 긴 HC 궤적(중앙값 32.5 step)에서 잘 작동하지 않는다.
4. **원인이 아님을 확인한 것**
   - agent 산출 방식: LLM이 출력한 `decisive_agent`(HC 46.6 / 51.7), `Orchestrator (-> X)`를 X로 간주(44.8 / 51.7), t̂ 기준 α̂(44.8 / 50.0) 모두 같거나 낮다.
   - 라벨 노이즈: 정답 agent ≠ 정답 step 발화자인 경우는 HC 3/58뿐이다.
   - 파싱 실패와 fallback: 최종 실행의 누락 예측은 0이고, fallback은 HC 3–5개다.
5. **확인할 수 없는 것**: 저자의 reasoning·temperature 설정, OpenRouter 경유 여부, 부록에 없는 후처리.

## 5. 해석할 때 주의할 점

### 5.1 TrajDebug `score_steps.py`와 지표 정의가 다르다
`score_steps.json`의 `step_loose1/2`는 **정답보다 0–1(0–2) assistant step 앞**만 인정하는 한쪽 방향 지표다. 논문의 허용 오차는 **|t̂ − t∗| ≤ k (양방향, 전체 step 기준)**다. 논문과 비교할 때는 `score_falat.json`을, TrajDebug 및 다른 baseline과 비교할 때는 `score_steps.json`을 쓴다. Exact step 정확도는 두 파일이 같다.

### 5.2 채점 대상: t̃ vs t̂
논문 §3(Metrics)은 t̂ / α̂로 적었고, §2.4는 Stage 4의 t̃ / ã를 검증된 답이라고 한다. `metrics`는 t̃, `metrics_stage3_t_hat`은 t̂ 기준이다. 차이는 최대 4%p다(AG medium step 45.2 vs 41.3).


## 6. 실행 기록

| 실험 | 실행 폴더 (`falat/outputs/…`) |
|---|---|
| HC 기본값 | `whoandwhen_handcrafted/openai-gpt-5.4-mini_20261001-151628` |
| HC medium | `whoandwhen_handcrafted/openai-gpt-5.4-mini_20261002-175601_reasoning-medium` |
| AG 기본값 | `whoandwhen_algorithm_generated/openai-gpt-5.4-mini_20261002-173307_stage4fix` |
| AG medium | `whoandwhen_algorithm_generated/openai-gpt-5.4-mini_20261002-182933_reasoning-medium` |

HC 기본값 실행은 Stage 4 수정 전에 돌렸지만, HC는 첫 agent step이 1이라 이 수정의 영향을 받지 않는다. 실행 결과(`falat/outputs/`)는 git에 올리지 않는다.

재현 명령:
```bash
bash run_pipeline_falat.sh                              # 기본값, 두 데이터셋
REASONING_EFFORT=medium bash run_pipeline_falat.sh      # reasoning medium
```
