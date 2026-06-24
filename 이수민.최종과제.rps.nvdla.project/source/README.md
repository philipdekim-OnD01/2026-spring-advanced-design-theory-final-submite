# RPS INT8 GPU vs NVDLA Experiment

## 목표

1. Macallan에서 DLA-friendly RPS CNN을 학습하고 full-integer INT8 TFLite와 ONNX를 export한다.
2. Jetson AGX Orin에서 native INT8 I/O TensorRT GPU engine과 strict DLA engine을 생성한다.
3. 동일한 test split으로 정확도와 end-to-end latency를 비교한다.
4. `tegrastats` compute rail을 이용해 GPU/DLA의 idle 보정 power를 비교한다.

## 모델 구조와 width

모델은 8개의 `Conv2D + BatchNorm + ReLU` block과 average pooling,
1x1 convolution classifier로 구성된다. 기본 출력 채널은 다음과 같다.

```text
[24, 32, 32, 48, 48, 64, 96, 128]
```

`width_mult`는 모든 block의 출력 채널 수를 곱하는 channel width 배율이다.
strict DLA 실험에서는 다음 식으로 각 채널을 32의 배수로 올림 정렬하였다.

```text
channels = max(32, ceil(base_channels * width_mult / 32) * 32)
```

layer 수, 입력 `64x64`, kernel `3x3`, stride 및 3개 출력 class는 고정했다.

## 1. 코드 위치

```bash
cd 이수민.최종과제.rps.nvdla.project/source
```

## 2. Dataset과 모델 다운로드

저장소 상위 폴더에서 실행한다.

```bash
./data/download_dataset.sh
./models/download_models.sh all
```

다운로드 후 다음 경로가 생성된다.

```text
source/RPS_Dataset/
source/artifacts/w0_25/
source/artifacts/w0_5/
source/artifacts/w1_0/
source/artifacts/w2_0/
source/artifacts/w4_0/
source/artifacts/w8_0/
source/artifacts/w16_0/
```

## 3. Macallan conda 환경

```bash
conda env create -f environment/macallan.yml
conda activate rps_nvdla_train
```

학습 예시:

```bash
python scripts/train_export_scaled_int8.py \
  --dataset-dir RPS_Dataset \
  --out-dir artifacts/w1_0 \
  --width-mult 1 \
  --channel-alignment 32 \
  --channel-rounding ceil \
  --epochs 40
```

## 4. Jetson conda 환경

JetPack과 TensorRT는 장치에 설치된 버전을 사용한다. 시스템 설정이나
`nvpmodel`은 변경하지 않는다.

```bash
conda env create -f environment/jetson.yml
conda activate rps_int8_nvdla
python -c 'import tensorrt, pycuda.driver; print(tensorrt.__version__)'
```

TensorRT import가 실패하면 JetPack에 포함된 Python wheel을 현재 conda
환경에 설치한다. 자세한 내용은 `environment/README.md`를 확인한다.

## 5. GPU와 strict DLA engine 생성 및 latency 측정

```bash
./scripts/run_agx_native_strict_sweep.sh \
  w0_25:w0_25 w0_5:w0_5 w1_0:w1_0 w2_0:w2_0 \
  w4_0:w4_0 w8_0:w8_0 w16_0:w16_0
```

각 DLA engine은 `--strict-dla`로 생성되어 GPU fallback을 허용하지 않는다.

## 6. Power sweep

```bash
./scripts/run_power_latency_sweep.sh
```

이 스크립트는 모델별 GPU/DLA를 각각 3회 실행한다. 측정 순서를
`GPU-DLA`, `DLA-GPU`, `GPU-DLA`로 교차하고 `tegrastats`를 200ms 간격으로
수집한다. 전력 mode와 clock은 변경하지 않는다.

## 7. 결과 집계

```bash
node scripts/summarize_power_latency_sweep.mjs \
  results/power_raw \
  ../results/native_strict_dla32_sweep_summary.csv \
  results/summary
```

## 8. Nsight profiling

```bash
BENCHMARK_SCRIPT=benchmark_trt_native_int8.py ./scripts/profile_nsight.sh \
  gpu_w1 engines/w1_0_gpu_native_int8_b1.engine \
  RPS_Dataset artifacts/w1_0/manifest.csv results/nsight_benchmark.csv \
  --calib-cache engines/w1_0_gpu_native_int8_b1.calib.cache \
  --device gpu --model-name rps_w1_gpu
```

`ncu`는 GPU CUDA kernel을 분석하는 도구이므로 strict DLA 내부 연산
kernel을 직접 분석하지는 못한다. DLA 경로는 Nsight Systems timeline과
TensorRT layer 정보를 중심으로 확인한다.
