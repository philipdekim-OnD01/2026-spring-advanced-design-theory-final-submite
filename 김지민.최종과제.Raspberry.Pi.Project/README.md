# 김지민 최종과제

## 프로젝트 제목

Raspberry Pi Project

## 제출 파일

- requirements-mac.txt
- requirements-pi.txt
- models/
- scripts/

## 최종 실습 결과 요약
- **목표**: Raspberry Pi 환경에서 가위바위보 Object Detection 모델 구동 및 INT8 양자화 최적화 성능 비교
- **모델 크기 감소율**: 원본 대비 **68.3% 용량 감소** (115,920 bytes -> 36,736 bytes)

## 하드웨어 벤치마크 결과 (Raspberry Pi)
- **FP32 SSD-Lite 추론 시간**: 평균 1.21 ms
- **INT8 SSD-Lite 추론 시간**: 평균 1.26 ms
- **메모리 사용량 (RSS)**: 두 모델 모두 약 42 MB 수준으로 동일하게 측정됨
  - *이유*: 모델 자체의 크기(수십 KB)에 비해 Python 환경, OpenCV, TFLite Runtime 라이브러리가 차지하는 기본 메모리 오버헤드가 더 크기 때문임을 확인.
