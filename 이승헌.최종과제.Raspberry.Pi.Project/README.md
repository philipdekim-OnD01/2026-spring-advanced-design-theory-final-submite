# RP2-RPS-QAT-Lab

## 프로젝트 제목

Rock-Paper-Scissors Classification on Raspberry Pi with QAT & Structured Pruning

## 프로젝트 개요

Raspberry Pi에 배포하기 위한 손 모양 인식(가위/바위/보) ResNet50 모델 최적화.
Quantization-Aware Training(QAT)과 Network Slimming(Structured Pruning)을 적용하여 정확도 유지 하에 모델 경량화.

## 제출 파일

### 노트북
- `notebooks/RPS_QAT_ResNet50_초보자용.ipynb` - 전체 실습 가이드

### 유틸리티
- `utils/pruning_ablation.py` - Structured pruning 구현
- `utils/structured_rebuild.py` - Tier3 모델 재구성

### 결과물 (results/)
- `.h5` 파일: Keras 모델 (float32)
- `.tflite` 파일: Raspberry Pi 배포용 (int8 양자화)

### 데이터셋
- `RPS_Dataset/` - 가위/바위/보 이미지 (0/1/2 폴더)

## 주요 기법

1. **QAT** - 양자화 손실 최소화
2. **L1 Sparsity Learning** - 채널별 중요도 학습  
3. **Tier1 vs Tier3** - 마스킹 vs 모델 재구성 비교
4. **TFLite 변환** - 모바일/임베디드 배포
