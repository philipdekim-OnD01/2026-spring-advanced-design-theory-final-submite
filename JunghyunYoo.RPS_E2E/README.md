# Junghyun Yoo Final Project

## RPS_E2E Project
Raspberry Pi 실시간 가위바위보 인식 시스템

개요
DenseNet121 기반 RPS 3-class 분류 모델을 transfer learning으로 학습하고, TFLite full-int8 모델로 변환한 뒤 Raspberry Pi 실시간 카메라 파이프라인에 배치하였다.
주요 실험은 baseline 학습, augmentation, PTQ/QAT 양자화, pruning/sparsity, camera-preprocess-inference-display latency 분석을 포함한다.

핵심 결과
Dataset: 2,717 images, train 2,172 / val 272 / test 273
Final model: DenseNet121 + strong augmentation + PTQ full-int8 TFLite
Test accuracy: 273 / 273 = 100.00%
Raspberry Pi inference: TFLite full-int8, TensorFlow/XNNPACK, 4 threads
Real-time pipeline: CSI camera, 640x480 dashboard

제출 파일 구성
source/: Raspberry Pi 최종 실시간 inference/dashboard 코드
models/: 최종 TFLite 모델
reports/: 주요 실험 보고서
figures/: 학습 그래프, latency breakdown, timing pie chart

최종 실행 코드
Raspberry Pi에서 다음 스크립트를 실행한다.
cd source
./RUN_FINAL_CSI_OV5647_COMPACT_DASHBOARD.sh
실제 dashboard application은 아래 파일이다.
source/pi_rps_window_dashboard_compact480.py
공통 camera/preprocess/inference logic은 아래 파일에 있다.
source/pi_realtime_rps.py

최종 모델
models/FINAL_best_strong_aug_ptq_full_int8_io.tflite
