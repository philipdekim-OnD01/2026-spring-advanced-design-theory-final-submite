# 김성주.최종과제.VLM.on.RSP

## 제출자

- 김성주
- GitHub 계정: `sungjukim99`

## 프로젝트 제목

VLM on Raspberry Pi — 웹캠 가위바위보 / 사물 인식

## 프로젝트 요약

Raspberry Pi에서 웹캠 프레임을 캡처하고, 로컬 `llama.cpp` 멀티모달 모델(SmolVLM-500M)에 보내 인식 결과를 출력하는 실습 프로젝트다. SSD 같은 전용 검출 모델 대신 범용 VLM(Vision-Language Model)을 온디바이스로 돌려, 별도 학습 없이 프롬프트만으로 가위바위보 손 모양과 사물을 인식한다.

처리 흐름은 다음과 같다.

1. 웹캠 프레임을 캡처한다.
2. 이미지를 로컬 `llama.cpp` 멀티모달 모델에 보낸다.
3. 인식 결과를 구조화된 형태로 출력한다.

두 가지 모드를 지원한다.

- `rps`: 가위(scissors), 바위(rock), 보(paper), 알 수 없음(unknown)으로 분류한다.
- `objects`: 화면에 보이는 주요 사물을 나열한다.

## 코드 구조

모든 스크립트는 이 폴더 안에 있고, 공통 로직은 하나의 헬퍼 모듈로 모았다.

```text
김성주.최종과제.VLM.on.RSP/
├── README.md
├── .gitignore                 # llama.cpp / 캡처 / 데이터셋 등 대용량 제외
├── vlm_common.py              # 공통 경로/프롬프트/캡처/명령 빌더/출력 파싱
├── camera_snapshot.py         # 카메라 점검용 단일 프레임 캡처
├── webcam_preview.py          # 실시간 웹캠 미리보기 창
├── vlm_webcam.py              # 프레임 캡처 후 배치 인식 (rps / objects)
├── rps_live_vlm.py            # 실시간 미리보기 + 주기적 VLM 결과 오버레이
├── start_500m_vlm_server.sh   # SmolVLM-500M llama-server 실행
├── setup_llama.sh             # llama.cpp 클론 + 빌드 (최초 1회)
├── captures/                  # 캡처된 프레임 저장 위치 (자동 생성, 비추적)
└── llama.cpp/                 # 추론 엔진 (repo에 미포함, setup_llama.sh로 생성)
```

`llama.cpp/`, `captures/`, `RPS_Dataset/`, `*.gguf`는 용량이 크거나 실행 중 생성되는 산출물이라 repo에 포함하지 않는다(`.gitignore`). 코드를 받은 뒤 아래 "사전 준비"로 직접 빌드한다.

## 사전 준비: llama.cpp 빌드

이 repo에는 `llama.cpp`가 포함되어 있지 않다. 최초 1회 아래 스크립트로 클론하고 빌드한다.

```bash
bash setup_llama.sh
```

스크립트는 `ggml-org/llama.cpp`를 클론한 뒤 CMake로 Release 빌드하고, `llama.cpp/build/bin/`에 `llama-mtmd-cli`와 `llama-server`를 만든다. 특정 버전을 고정하려면 환경 변수로 지정한다.

```bash
LLAMA_REF=b4000 bash setup_llama.sh
```

빌드에는 `git`, `cmake`, C/C++ 컴파일러가 필요하다. Raspberry Pi 기준:

```bash
sudo apt install -y git cmake build-essential
```

## 현재 로컬 상태

- 웹캠 장치가 `/dev/video0`에 있다.
- Python OpenCV가 설치되지 않은 환경에서는 스크립트가 `ffmpeg`로 대체 캡처한다.
- VLM GGUF 모델은 Hugging Face에서 자동 다운로드한다(`--hf-repo`).

## 카메라 테스트

```bash
python3 camera_snapshot.py --camera /dev/video0 --output captures/test.jpg
```

`/dev/video0`이 동작하지 않으면 `/dev/video1`을 시도한다.

## 실시간 웹캠 미리보기

headless SSH 셸이 아니라 Raspberry Pi 데스크톱 터미널에서 실행한다.

```bash
python3 webcam_preview.py --camera /dev/video0
```

전체 화면:

```bash
python3 webcam_preview.py --camera /dev/video0 --fullscreen
```

미리보기 창에서 `q`를 누르면 종료된다.

## 가위바위보 인식 실행

`llama.cpp`가 지원하는 Hugging Face GGUF VLM repo를 사용한다.

```bash
python3 vlm_webcam.py \
  --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF \
  --mode rps \
  --once
```

연속 인식:

```bash
python3 vlm_webcam.py \
  --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF \
  --mode rps \
  --interval 2
```

## 실시간 가위바위보 인식

Raspberry Pi 모니터에 웹캠 영상을 띄우고, 실행 중 VLM 결과를 갱신해서 보여준다.

터미널 1에서 500M VLM 서버를 계속 실행해 둔다.

```bash
bash start_500m_vlm_server.sh
```

터미널 2에서 실시간 미리보기를 실행한다.

```bash
python3 rps_live_vlm.py \
  --server-url http://127.0.0.1:8082 \
  --display :0 \
  --interval 2 \
  --crop-scale 0.55 \
  --infer-width 160
```

노란색 박스가 VLM에 전달되는 영역이다. 속도를 더 끌어올리려면 crop을 작게, 추론 입력 너비를 낮춘다.

```bash
python3 rps_live_vlm.py \
  --server-url http://127.0.0.1:8082 \
  --display :0 \
  --interval 2 \
  --crop-scale 0.35 \
  --infer-width 96
```

손이 가운데에 있지 않으면 crop 박스를 이동한다.

```bash
python3 rps_live_vlm.py \
  --server-url http://127.0.0.1:8082 \
  --display :0 \
  --crop-offset-x -0.4 \
  --crop-offset-y 0.2
```

미리보기 창에서 `q`를 누르면 종료된다. 모델이 처음 워밍업하는 동안 첫 결과는 몇 초 걸릴 수 있다.

## 사물 인식 실행

```bash
python3 vlm_webcam.py \
  --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF \
  --mode objects \
  --once
```

## 로컬 모델 파일 사용

모델과 프로젝터가 이미 다운로드되어 있으면 직접 지정한다.

```bash
python3 vlm_webcam.py \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --mode rps \
  --once
```

## OpenCV 설치 (선택)

웹캠을 반복적으로 읽을 때 더 빠르다.

```bash
sudo apt install python3-opencv
```

OpenCV가 없어도 스크립트는 매 프레임마다 `ffmpeg`를 호출해 동작한다.

## 문제 해결

카메라가 안 보이면 장치를 확인한다.

```bash
ls -l /dev/video*
```

`llama-mtmd-cli`를 못 찾으면 먼저 빌드되어 있는지 확인하고, 없으면 `setup_llama.sh`를 실행한다.

```bash
ls llama.cpp/build/bin/llama-mtmd-cli || bash setup_llama.sh
```

SSH에서 미리보기 창이 안 뜨면 `--display :0`을 지정하고 Raspberry Pi 모니터를 확인한다. 기본적으로 스크립트는 `DISPLAY`와 `~/.Xauthority`를 자동으로 잡는다.
