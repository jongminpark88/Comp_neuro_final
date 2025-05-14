# Comp_neuro_final

## format
# 📁 Comp_neuro_final

## 🔧 루트 디렉토리
- `README.md` : 프로젝트 개요, 설치 및 실행 방법
- `requirements.txt` : 필요한 Python 패키지 목록
- `setup.py` : (선택) 패키징 및 설치용 스크립트

---

## 📁 envs/
- `custom_maze_env.py` : MapEnv를 상속받은 커스텀 미로 환경 정의

---

## 📁 src/ (핵심 모델 구성)
### 📁 memory/
- `slots.py` : (obs, Δt, h) 형태의 메모리 슬롯 클래스
- `bank.py` : 메모리 뱅크 구조 및 접근 인터페이스
- `forget.py` : 망각곡선 기반 노이즈 추가 로직

### 📁 retrieval/
- `attention.py` : 유사도 계산 및 어텐션 모듈
- `gate.py` : 회상 게이트 메커니즘

### 📁 model/
- `rnn_agent.py` : RNN + 회상 통합 에이전트 모델 정의

### 📁 utils/
- `logger.py` : 학습 로그 및 TensorBoard 유틸리티
- `metrics.py` : 평가 지표 정의 (성공률, consistency 등)

---

## 📁 experiments/ (학습 및 평가 스크립트)
### 📁 configs/
- `default.yaml` : 기본 실험 설정
- `debug.yaml` : 디버깅용 소규모 설정

- `train.py` : 학습 루프 실행 스크립트
- `test.py` : 평가/검증용 실행 스크립트
- `analyze.py` : 결과 분석 및 시각화 스크립트

---

## 📁 notebooks/
- `01_data_inspection.ipynb` : 저장된 메모리 구조 확인
- `02_gate_behavior.ipynb` : 게이트 값 시각화 및 회상 분석
- `03_performance_comparison.ipynb` : 성능 지표 비교 분석

---

## 📁 results/
- `logs/` : TensorBoard 로그 파일
- `checkpoints/` : 학습된 모델 체크포인트
- `figures/` : 학습 곡선, attention 분포 등 시각화 결과

---

## 📁 scripts/
- `run_all.sh` : 전체 실험 일괄 실행 스크립트
- `env_setup.sh` : 가상 환경 설정 및 패키지 설치 스크립트
