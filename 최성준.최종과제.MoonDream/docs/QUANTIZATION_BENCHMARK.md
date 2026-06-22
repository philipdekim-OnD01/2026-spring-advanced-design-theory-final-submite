# Moondream Model Quantization Benchmark

## Overview

This document provides comprehensive documentation for benchmarking the Moondream vision-language model with various quantization techniques. Quantization is a technique to reduce model size and memory usage by using lower-precision data types for model weights and activations.

## Table of Contents

1. [What is Quantization?](#what-is-quantization)
2. [Quantization Methods Tested](#quantization-methods-tested)
3. [Installation Requirements](#installation-requirements)
4. [Usage Guide](#usage-guide)
5. [Metrics Explained](#metrics-explained)
6. [Expected Results](#expected-results)
7. [Recommendations](#recommendations)
8. [Troubleshooting](#troubleshooting)

---

## What is Quantization?

**Quantization** is a model compression technique that reduces the precision of the numbers used to represent model parameters. Instead of using 32-bit or 16-bit floating-point numbers, quantization uses lower precision formats like 8-bit or 4-bit integers.

### Benefits of Quantization:
- **Reduced Memory Usage**: 4-bit quantization can reduce model size by ~75% (4x compression)
- **Faster Inference**: Lower precision operations can be faster on compatible hardware
- **Lower Storage Requirements**: Smaller model files for deployment
- **Edge Device Deployment**: Enables running large models on resource-constrained devices

### Trade-offs:
- **Slight Accuracy Loss**: Lower precision may reduce output quality slightly
- **Compatibility**: Requires specific hardware/software support (e.g., CUDA, bitsandbytes)

---

## Quantization Methods Tested

### 1. Baseline (FP16)
- **Precision**: 16-bit floating-point
- **Purpose**: Reference for comparison
- **Characteristics**: Full model quality, highest memory usage
- **Use Case**: When maximum quality is required and memory is not constrained

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 2. 8-bit Quantization
- **Precision**: 8-bit integers
- **Compression**: ~2x memory reduction
- **Quality**: Minimal quality degradation
- **Use Case**: Good balance between quality and efficiency

```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto"
)
```

### 3. 4-bit NF4 Quantization (Recommended)
- **Precision**: 4-bit Normal Float (NF4)
- **Compression**: ~4x memory reduction
- **Quality**: Optimized for normal distribution of weights
- **Use Case**: Maximum memory savings with acceptable quality

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto"
)
```

**NF4 Parameters Explained:**
- `load_in_4bit=True`: Enable 4-bit quantization
- `bnb_4bit_compute_dtype=torch.float16`: Use FP16 for computation (accuracy vs speed)
- `bnb_4bit_quant_type="nf4"`: Use Normal Float 4-bit (optimized for neural networks)
- `bnb_4bit_use_double_quant=True`: Apply nested quantization for additional memory savings

### 4. 4-bit FP4 Quantization
- **Precision**: 4-bit Floating Point (FP4)
- **Compression**: ~4x memory reduction
- **Quality**: Alternative to NF4, may perform differently on specific models
- **Use Case**: Compare against NF4 to find best 4-bit configuration

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="fp4",  # FP4 instead of NF4
    bnb_4bit_use_double_quant=True
)
```

---

## Installation Requirements

### Required Libraries

```bash
# Core dependencies
pip install torch torchvision
pip install transformers
pip install pillow

# For quantization
pip install bitsandbytes
pip install accelerate

# For system monitoring
pip install psutil
```

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for quantization)
- **CUDA**: Version 11.0 or higher
- **RAM**: At least 16GB recommended
- **VRAM**: At least 4GB for baseline model, 1-2GB for quantized models

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## Usage Guide

### Basic Usage

1. **Prepare Test Image**: Ensure you have a test image available

2. **Update Configuration** (if needed):
   ```python
   # Edit in benchmark_quantization.py
   TEST_IMAGE_PATH = "path/to/your/test/image.png"
   TEST_QUESTION = "Please describe in detail"
   ```

3. **Run Benchmark**:
   ```bash
   python benchmark_quantization.py
   ```

### Configuration Options

You can modify these parameters in `benchmark_quantization.py`:

```python
MODEL_ID = "vikhyatk/moondream2"           # Model to benchmark
TEST_IMAGE_PATH = "path/to/image.png"      # Test image
TEST_QUESTION = "Please describe in detail" # Question to ask
NUM_WARMUP_RUNS = 2                         # Warmup iterations
NUM_TEST_RUNS = 5                           # Benchmark iterations
```

### Custom Quantization Configuration

To test custom quantization settings, add a new function:

```python
def test_custom_quantization():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Try BF16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False  # Disable double quant
    )

    # ... rest of the code
```

---

## Metrics Explained

### 1. Model Size
- **What**: Total size of model parameters and buffers
- **Unit**: Megabytes (MB)
- **Importance**: Determines storage and memory requirements
- **Expected**: 4-bit ≈ 1/4 of baseline, 8-bit ≈ 1/2 of baseline

### 2. GPU Memory Usage
- **What**: GPU VRAM consumed during inference
- **Unit**: Megabytes (MB)
- **Importance**: Critical for determining if model fits on GPU
- **Measured**: Before and after inference (delta shows peak usage)

### 3. CPU Memory Usage
- **What**: System RAM consumed by the process
- **Unit**: Megabytes (MB)
- **Importance**: Relevant for CPU-only deployments

### 4. Inference Time
- **What**: Time to process one image + generate answer
- **Unit**: Milliseconds (ms)
- **Statistics**:
  - **Average**: Mean time across test runs
  - **Min**: Fastest run (best-case scenario)
  - **Max**: Slowest run (worst-case scenario)
- **Note**: Lower precision may sometimes be faster, but not guaranteed

### 5. Parameter Count
- **Total Parameters**: All model weights
- **Trainable Parameters**: Parameters that can be updated during training
- **Note**: Count stays the same, only precision changes

---

## Expected Results

### Typical Performance Profile

| Configuration | Model Size | GPU Memory | Inference Time | Quality |
|--------------|------------|------------|----------------|---------|
| Baseline FP16 | ~3500 MB | ~4000 MB | ~2000 ms | Excellent |
| 8-bit | ~1800 MB | ~2200 MB | ~2100 ms | Very Good |
| 4-bit NF4 | ~900 MB | ~1200 MB | ~2200 ms | Good |
| 4-bit FP4 | ~900 MB | ~1200 MB | ~2200 ms | Good |

*Note: Actual numbers depend on your hardware and model version*

### Sample Benchmark Output

```
================================================================================
COMPARISON SUMMARY
================================================================================

Configuration             Model Size      GPU Memory      Avg Time
                          (MB)            (MB)            (ms)
--------------------------------------------------------------------------------
Baseline (FP16)             3472.54 MB     3856.32 MB     1987.45 ms
8-bit Quantization          1789.23 MB     2034.67 MB     2043.12 ms
  vs Baseline                   1.94x          1.89x          1.03x
4-bit NF4 Quantization       894.67 MB     1156.89 MB     2134.56 ms
  vs Baseline                   3.88x          3.33x          1.07x
4-bit FP4 Quantization       896.34 MB     1159.23 MB     2128.91 ms
  vs Baseline                   3.87x          3.33x          1.07x
```

### Memory Savings Explained

- **4-bit**: ~75% reduction (4x compression)
- **8-bit**: ~50% reduction (2x compression)
- **Double quantization**: Additional ~5-10% savings

---

## Recommendations

### Production Deployment

#### For Edge Devices (Raspberry Pi, Jetson Nano, Mobile)
**Recommendation: 4-bit NF4**
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```
- **Why**: Maximum memory savings, enables deployment on low-VRAM devices
- **Trade-off**: Slight quality reduction, acceptable for most applications

#### For Cloud/Server Deployment
**Recommendation: 8-bit**
```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```
- **Why**: Best quality-to-efficiency ratio, 2x memory savings
- **Trade-off**: Minimal quality loss, good cost savings

#### For Research/Development
**Recommendation: Baseline FP16 or FP32**
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16  # or torch.float32
)
```
- **Why**: Maximum accuracy for evaluation and fine-tuning
- **Trade-off**: Higher memory requirements

### Quality vs Efficiency Trade-off

```
Quality:    FP32 > FP16 > 8-bit > 4-bit NF4 ≈ 4-bit FP4
Efficiency: 4-bit NF4 > 4-bit FP4 > 8-bit > FP16 > FP32
```

### When to Choose Each Method

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Limited VRAM (<4GB) | 4-bit NF4 | Only option that fits |
| Mobile deployment | 4-bit NF4 | Smallest size, fast enough |
| Batch processing | 8-bit | Good throughput, quality |
| Real-time critical | Baseline FP16 | Lowest latency |
| Quality critical | Baseline FP16/32 | Best accuracy |
| Cost optimization | 8-bit | Balance of all factors |

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size (if processing multiple images)
- Use more aggressive quantization (4-bit instead of 8-bit)
- Clear GPU cache: `torch.cuda.empty_cache()`
- Close other GPU-intensive applications

#### 2. bitsandbytes Not Found
```
ModuleNotFoundError: No module named 'bitsandbytes'
```
**Solution**:
```bash
pip install bitsandbytes
```
For Windows users, use Windows Subsystem for Linux (WSL) or use pre-built wheels.

#### 3. Slow Inference on CPU
```
Inference taking 10x+ longer than expected
```
**Solution**:
- Quantization requires CUDA for efficiency
- Install CUDA-compatible PyTorch
- Verify GPU is being used: `torch.cuda.is_available()`

#### 4. Model Loading Fails
```
OSError: Can't load model
```
**Solution**:
- Check internet connection (first load downloads model)
- Verify model ID is correct: `"vikhyatk/moondream2"`
- Use `trust_remote_code=True` parameter

#### 5. Different Results Each Run
**Cause**: Normal variation due to GPU scheduling and thermal throttling

**Solution**:
- Increase `NUM_TEST_RUNS` for more stable averages
- Ensure adequate GPU cooling
- Run warmup iterations before benchmarking

---

## Advanced Topics

### Custom Quantization for Fine-tuning

If you plan to fine-tune the quantized model (QLoRA):

```python
from peft import prepare_model_for_kbit_training

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
```

### Mixed Precision Inference

For even better efficiency:

```python
with torch.cuda.amp.autocast():
    encoded_image = model.encode_image(image)
    answer = model.answer_question(encoded_image, question, tokenizer)
```

### Profiling Memory Usage

For detailed memory profiling:

```python
import torch.cuda.memory as cuda_memory

# Before inference
torch.cuda.reset_peak_memory_stats()

# Your inference code
with torch.no_grad():
    output = model(input)

# After inference
peak_memory = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU memory: {peak_memory:.2f} MB")
```

---

## References

- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [Moondream Model](https://huggingface.co/vikhyatk/moondream2)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

---

## Benchmark Script Details

### File: `benchmark_quantization.py`

**Key Functions:**
- `benchmark_model()`: Core benchmarking logic
- `test_baseline()`: FP16 baseline test
- `test_8bit_quantization()`: 8-bit quantization test
- `test_4bit_quantization()`: 4-bit NF4 test
- `test_4bit_fp4_quantization()`: 4-bit FP4 test
- `print_comparison_table()`: Results summary

**Customization Points:**
- Modify `NUM_WARMUP_RUNS` for stability
- Modify `NUM_TEST_RUNS` for accuracy
- Change `TEST_IMAGE_PATH` for different inputs
- Add custom quantization configurations

---

## Conclusion

Quantization is a powerful technique for deploying large vision-language models on resource-constrained devices. This benchmark helps you:

1. **Measure** actual memory and speed improvements
2. **Compare** different quantization methods objectively
3. **Choose** the right configuration for your use case
4. **Validate** that quantization maintains acceptable quality

**Quick Decision Guide:**
- Need maximum quality? → Use **Baseline FP16**
- Need balanced performance? → Use **8-bit**
- Need maximum efficiency? → Use **4-bit NF4**
- Unsure? → **Run the benchmark and compare!**

---

**Last Updated**: 2026-06-15
**Version**: 1.0
**Author**: Quantization Benchmark Tool
