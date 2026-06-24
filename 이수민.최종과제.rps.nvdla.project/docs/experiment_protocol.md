# Experiment protocol

## Hardware and software

- Training: Macallan server, TensorFlow 2.15.1
- Deployment: Jetson AGX Orin
- TensorRT: 8.6.2
- Precision: INT8 PTQ
- Input: batch 1, `3x64x64`
- DLA: DLA0, strict mode, GPU fallback disabled
- Power mode: observed MAXN; no setting changed

## Model sweep

Eight convolution blocks were held constant while output channels were scaled
with width multipliers `0.25, 0.5, 1, 2, 4, 8, 16`. Channels were rounded up
to multiples of 32 for DLA alignment.

## Accuracy and latency

- Fixed class-stratified train/validation/test manifest
- Test set: 543 images
- Warmup: 100 inferences
- Per run: 3 passes over the test set, 1,629 timed inferences
- Independent runs: 5 per model and device
- Timing: H2D, TensorRT execute, D2H and stream synchronization

## Power

- Source: `tegrastats`
- Interval: 200ms
- Rail: `VDD_GPU_SOC + VDD_CPU_CV`
- Idle baseline: measured immediately before each load
- Load: one TensorRT stream, transfer-disabled continuous saturation
- Warmup/load: 3s + 12s
- Repeats: 3 per model/device, alternating execution order
- Dynamic power: load compute-rail power minus preceding idle power

Power measurement isolates accelerator execution. It is not total wall power.
