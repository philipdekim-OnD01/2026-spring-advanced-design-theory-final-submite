# Environment setup

모든 Python dependency는 사용자 conda 환경에 설치한다. Jetson의 전역
Python, CUDA, TensorRT, power mode 및 system environment는 변경하지 않는다.

## Macallan

```bash
conda env create -f macallan.yml
conda activate rps_nvdla_train
```

## Jetson AGX Orin

```bash
conda env create -f jetson.yml
conda activate rps_int8_nvdla
```

JetPack의 TensorRT wheel 위치는 release에 따라 다르다. 다음으로 확인한다.

```bash
find /usr -path '*tensorrt*' -name 'tensorrt-*-cp310-*.whl' 2>/dev/null
```

wheel이 발견되면 현재 conda 환경에만 설치한다.

```bash
python -m pip install /usr/src/tensorrt/python/tensorrt-*-cp310-*.whl
```

마지막으로 다음 import를 확인한다.

```bash
python -c 'import tensorrt, pycuda.driver; print(tensorrt.__version__)'
```
