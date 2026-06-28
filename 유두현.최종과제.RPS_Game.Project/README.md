# RPS_SSD_WINNER_DETECTOR

Raspberry Pi에서 가위바위보 손 모양을 SSD Object Detection 방식으로 실시간 탐지하고, 여러 플레이어의 텐서 출력값을 연산하여 승패(WINNER/LOSE/DRAW)를 실시간으로 가려내는 온디바이스 AI 융합 어플리케이션 실습 레포지토리다.

## 실습 목표

* Raspberry Pi에서 `RPS_PreTrained_SSD.tflite`를 이용해 실시간 다중 손 모양 탐지를 실행한다.
* 딥러닝 모델의 출력 텐서(Bounding Box, Class ID, Confidence Score)를 파싱하는 후처리 파이프라인을 이해한다.
* 규칙 기반의 가위바위보 게임 엔진 알고리즘 함수를 구현하여 객체 탐지 시스템과 결합한다.
* Raspberry Pi 하드웨어 환경에서 FP32 및 INT8 양자화 경량화 모델 간의 자원 소모량과 추론 Latency 성능을 벤치마크한다.

## 폴더 구조

```text
RSP_SSD_LAB/
├── README.md
├── requirements-mac.txt
├── requirements-pi.txt
├── assets/
│   └── representative_images/
├── docs/
│   └── 라즈베리파이_SSD_오브젝트디텍션_실행매뉴얼.md
├── models/
│   ├── RPS_PreTrained_SSD.tflite          # 실시간 가위바위보 게임 구동용 핵심 모델
│   ├── ssd_lite_rps.h5
│   ├── rps_ssd_lite_fp32.tflite           # 벤치마크 비교용 FP32 모델
│   └── rps_ssd_lite_int8.tflite           # 벤치마크 비교용 INT8 양자화 모델
└── scripts/
    ├── run_rps_ssd_camera.py
    ├── EX_03_Board_RPS_PreTrained_SSD.py  
    ├── RPS_Game_Test.py  # [핵심] 가위바위보 승패 판정 알고리즘이 구현된 메인 스크립트
    ├── EX_01_Image_Capture.py
    ├── quantize_rps_ssd_lite.py
    └── benchmark_rps_ssd_lite_tflite.py

```

## 핵심 기능 및 알고리즘 구현 개요

기존의 객체 탐지가 단순히 손의 위치를 상자로 시각화하는 것에 그쳤다면, 본 프로젝트는 다차원 출력 데이터를 바탕으로 판단 및 액션을 취하는 애플리케이션 레이어를 통합(S/W-H/W Co-design)한 점이 핵심이다.

### 1. 가위바위보 상성 판정 및 승자 추출 알고리즘 (`EX_03_Board_RPS_PreTrained_SSD.py`)

* 모델이 예측한 `class_index`를 추출하여 상성을 비교 연산하는 규칙 엔진(`judge_rules`)을 구축했다.
* `threshold = 0.8` 이상으로 검출된 유효 객체가 **정확히 2개**인 프레임을 실시간 필터링하여 플레이어 간의 승패를 판정한다.

### 2. 동적 시각화 후처리 융합

* **WINNER (승리):** 상자가 **초록색**으로 변경되며 클래스명 앞에 `[WINNER]` 텍스트가 바인딩된다.
* **LOSE (패배):** 상자가 **빨간색**으로 변경되며 클래스명 앞에 `[LOSE]` 텍스트가 바인딩된다.
* **DRAW (무승부):** 같은 모양을 내어 비겼을 경우 상자가 **파란색**으로 갱신되며 `[DRAW]` 텍스트가 표시된다.
* 단일 손이거나 3개 이상의 객체가 잡힐 때는 판정을 유보하고 기존의 기본 바운딩 박스를 렌더링한다.

---

## 실행 및 구동 방법

### 1. 하드웨어 가상환경 검증 (Raspberry Pi)

가상환경 내 배포용 패키지가 정상 동작하는지 터미널에서 사전 확인한다.

```bash
cd ~/camera_test/cnn/examples/05_Object_Detection_Based_On-Device_AI
~/camera_test/.venv311/bin/python - <<'PY'
import importlib.util
for m in ["cv2", "numpy", "tflite_runtime"]:
    print(m, bool(importlib.util.find_spec(m)))
PY

```

### 2. 가위바위보 승자 디텍터 구동

라즈베리 파이 로컬 터미널 혹은 디스플레이가 연결된 환경에서 승패 판정 알고리즘 스크립트를 다이렉트로 실행한다.

```bash
cd ~/camera_test/cnn/examples/05_Object_Detection_Based_On-Device_AI
~/camera_test/.venv311/bin/python RPS_Game_Test.py

```

* **원격 MacBook SSH 접속 환경**에서 라즈베리파이 로컬 화면(:0)으로 카메라 창을 강제 포워딩하여 띄울 시:
```bash
env DISPLAY=:0 XAUTHORITY=/home/doohyun/.Xauthority ~/camera_test/.venv311/bin/python EX_03_Board_RPS_PreTrained_SSD.py

```



---

## 모델 최적화 및 양자화 벤치마크 결과

하드웨어 제약 조건을 극복하기 위해 `scripts/benchmark_rps_ssd_lite_tflite.py`를 활용하여 측정한 정수 양자화(Post-Training Quantization) 전후의 성능 지표는 다음과 같다.

| 평가 지표 | FP32 SSD-Lite 모델 | INT8 양자화 모델 | 하드웨어 개선 효과 |
| --- | --- | --- | --- |
| **파일 크기** | 115,876 bytes (약 113KB) | 36,768 bytes (약 36KB) | **68.3% 용량 경량화** |
| **평균 추론 시간** | 1.89 ms | 1.39 ms | **약 1.36배 연산 가속** |
| **`ps` CPU 사용률** | 172.0% | 148.0% | **CPU 부하 감소 (24%p↓)** |
| **`ps` RSS 메모리** | 42.17 MB | 41.97 MB | 변동 미미 (약 0.2MB 감소) |

### 💡 벤치마크 결과 해석

1. **연산 가속 및 CPU 사용률 감소 원인:** ARM 기반 임베디드 CPU 아키텍처는 부동소수점(`float32`) 연산 장치보다 정수(`int8`) 연산을 처리할 때 하드웨어 클록 주기를 훨씬 적게 소모하므로 속도가 향상되고 연산 스트레스가 감소한다.
2. **RSS 메모리(실제 RAM 점유)가 수십 MB대로 유지되는 이유:** 경량화 과정을 거쳐 모델의 용량은 수십 KB 수준으로 압축되었으나, 프로그램을 구동하기 위한 인프라 스택(Python 인터프리터, OpenCV 카메라 드라이버, NumPy 행렬 라이브러리 및 TFLite Runtime)이 차지하는 고유의 기본 시스템 메모리 풋프린트가 크기 때문이다.
3. **최종 통찰:** 엣지 단의 피지컬 AI를 효과적으로 서빙하기 위해서는 네트워크 가중치의 경량화뿐만 아니라 전체 구동 소프트웨어 런타임의 경량화가 수반되어야 함을 증명한다.

---