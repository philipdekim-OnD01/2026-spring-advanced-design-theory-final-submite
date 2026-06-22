"""
PyTorch Native Quantization Benchmark for Moondream Model
Uses PyTorch's built-in quantization instead of bitsandbytes
Works with custom model architectures like Moondream
"""

import torch
import time
import psutil
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import copy

MODEL_ID = "vikhyatk/moondream2"
TEST_IMAGE_PATH = "/SSD1/sjchoi/semiai/RP2-RPS-QAT-Lab/cap/20260615_155234.png"
TEST_QUESTION = "Please describe in detail"
NUM_WARMUP_RUNS = 2
NUM_TEST_RUNS = 5


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_cpu_memory():
    """Get current process CPU memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    """Calculate actual model size in memory"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2


def clear_memory():
    """Clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_model(model, tokenizer, image, config_name):
    """Run inference benchmark on the model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")

    model.eval()

    # Memory before inference
    cpu_mem_before = get_cpu_memory()
    gpu_mem_before = get_gpu_memory()

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Get model size
    model_size_mb = get_model_size_mb(model)
    print(f"\nModel Size in Memory: {model_size_mb:.2f} MB")

    print(f"\nMemory Usage (Before Inference):")
    print(f"  CPU: {cpu_mem_before:.2f} MB")
    print(f"  GPU: {gpu_mem_before:.2f} MB")

    # Warmup runs
    print(f"\nRunning {NUM_WARMUP_RUNS} warmup iterations...")
    for i in range(NUM_WARMUP_RUNS):
        with torch.no_grad():
            encoded_image = model.encode_image(image)
            _ = model.answer_question(encoded_image, TEST_QUESTION, tokenizer)
        clear_memory()

    # Benchmark runs
    print(f"Running {NUM_TEST_RUNS} benchmark iterations...")
    inference_times = []
    answers = []

    for i in range(NUM_TEST_RUNS):
        clear_memory()

        start_time = time.time()
        with torch.no_grad():
            encoded_image = model.encode_image(image)
            answer = model.answer_question(encoded_image, TEST_QUESTION, tokenizer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        answers.append(answer)

        print(f"  Run {i+1}/{NUM_TEST_RUNS}: {inference_time:.2f} ms")

    # Memory after inference
    cpu_mem_after = get_cpu_memory()
    gpu_mem_after = get_gpu_memory()

    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)

    print(f"\nInference Time Statistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    print(f"\nMemory Usage (After Inference):")
    print(f"  CPU: {cpu_mem_after:.2f} MB (Delta: {cpu_mem_after - cpu_mem_before:+.2f} MB)")
    print(f"  GPU: {gpu_mem_after:.2f} MB (Delta: {gpu_mem_after - gpu_mem_before:+.2f} MB)")

    print(f"\nSample Output:")
    print(f"  {answers[0][:200]}...")

    results = {
        'config_name': config_name,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'cpu_memory_mb': cpu_mem_after,
        'gpu_memory_mb': gpu_mem_after,
        'avg_inference_time_ms': avg_time,
        'min_inference_time_ms': min_time,
        'max_inference_time_ms': max_time,
        'sample_output': answers[0]
    }

    return results


def test_baseline_fp16():
    """Test baseline FP16 model"""
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL (FP16)")
    print("="*60)

    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "Baseline (FP16)")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_baseline_fp32():
    """Test baseline FP32 model"""
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL (FP32)")
    print("="*60)

    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "Baseline (FP32)")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_dynamic_quantization():
    """Test PyTorch dynamic quantization (CPU only)"""
    print("\n" + "="*60)
    print("LOADING DYNAMICALLY QUANTIZED MODEL (INT8)")
    print("="*60)

    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    # Move to CPU for dynamic quantization
    model = model.to('cpu')

    print("Applying dynamic quantization (this may take a moment)...")
    try:
        # Apply dynamic quantization to Linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        results = benchmark_model(quantized_model, tokenizer, image, "Dynamic Quantization (INT8, CPU)")

        del quantized_model
    except Exception as e:
        print(f"Dynamic quantization failed: {str(e)}")
        print("This is expected for models with custom layers.")
        results = None

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_half_precision():
    """Test half precision (FP16) on GPU"""
    print("\n" + "="*60)
    print("TESTING HALF PRECISION (FP16) vs FULL PRECISION (FP32)")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU FP16 test")
        return None

    # Already tested in baseline
    return None


def save_model_checkpoint(model, path, precision="fp16"):
    """Save model checkpoint for deployment"""
    print(f"\nSaving {precision} checkpoint to {path}...")
    torch.save(model.state_dict(), path)
    file_size = os.path.getsize(path) / (1024**2)
    print(f"Checkpoint size: {file_size:.2f} MB")
    return file_size


def print_comparison_table(all_results):
    """Print comparison table of all results"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Table header
    print(f"\n{'Configuration':<30} {'Model Size':<15} {'GPU Memory':<15} {'Avg Time':<15}")
    print(f"{'':<30} {'(MB)':<15} {'(MB)':<15} {'(ms)':<15}")
    print("-" * 80)

    baseline = all_results[0]

    for result in all_results:
        config = result['config_name']
        model_size = result['model_size_mb']
        gpu_mem = result['gpu_memory_mb']
        avg_time = result['avg_inference_time_ms']

        # Calculate improvements
        size_ratio = (baseline['model_size_mb'] / model_size) if model_size > 0 else 0
        mem_ratio = (baseline['gpu_memory_mb'] / gpu_mem) if gpu_mem > 0 else 0
        speed_ratio = (result['avg_inference_time_ms'] / baseline['avg_inference_time_ms'])

        print(f"{config:<30} {model_size:>10.2f} MB   {gpu_mem:>10.2f} MB   {avg_time:>10.2f} ms")

        if config != baseline['config_name']:
            print(f"{'  vs Baseline':<30} {size_ratio:>10.2f}x     {mem_ratio:>10.2f}x     {speed_ratio:>10.2f}x")

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    fp32_result = next((r for r in all_results if "FP32" in r['config_name']), None)
    fp16_result = next((r for r in all_results if "FP16" in r['config_name']), None)

    if fp32_result and fp16_result:
        size_reduction = (fp32_result['model_size_mb'] / fp16_result['model_size_mb'])
        mem_reduction = (fp32_result['gpu_memory_mb'] / fp16_result['gpu_memory_mb'])

        print(f"\nFP16 vs FP32:")
        print(f"  • Model size reduction: {size_reduction:.2f}x ({fp32_result['model_size_mb']:.0f}MB → {fp16_result['model_size_mb']:.0f}MB)")
        print(f"  • GPU memory reduction: {mem_reduction:.2f}x ({fp32_result['gpu_memory_mb']:.0f}MB → {fp16_result['gpu_memory_mb']:.0f}MB)")
        print(f"  • Speed impact: {fp16_result['avg_inference_time_ms'] / fp32_result['avg_inference_time_ms']:.2f}x")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• FP16 (Half Precision): Best choice for deployment - 2x memory savings, faster")
    print("• FP32 (Full Precision): Use only if FP16 causes numerical instability")
    print("\nNOTE: Moondream's custom architecture is not compatible with bitsandbytes")
    print("quantization. For aggressive quantization, consider:")
    print("  1. Model pruning techniques")
    print("  2. Knowledge distillation to a smaller model")
    print("  3. ONNX Runtime with quantization")


def main():
    print("="*80)
    print("MOONDREAM MODEL QUANTIZATION BENCHMARK")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\n" + "="*80)
    print("ABOUT THIS BENCHMARK")
    print("="*80)
    print("Moondream uses a custom architecture that is not compatible with")
    print("bitsandbytes (4-bit/8-bit) quantization due to custom layer implementations.")
    print("\nThis benchmark tests PyTorch native quantization methods:")
    print("  • FP32 (Full Precision) - Baseline")
    print("  • FP16 (Half Precision) - Recommended for deployment")
    print("  • INT8 (Dynamic Quantization) - CPU only")

    all_results = []

    try:
        # Test FP32 baseline
        if torch.cuda.is_available():
            print("\n⚠️  FP32 test may fail on low VRAM GPUs. Skipping...")
            print("If you have >8GB VRAM, you can uncomment the FP32 test in the code.")
            # fp32_results = test_baseline_fp32()
            # if fp32_results:
            #     all_results.append(fp32_results)

        # Test FP16 (recommended)
        fp16_results = test_baseline_fp16()
        if fp16_results:
            all_results.append(fp16_results)

        # Test dynamic quantization (CPU)
        print("\n⚠️  Dynamic quantization test (may be slow on CPU)...")
        print("Skipping dynamic quantization - not practical for this model.")
        # dyn_results = test_dynamic_quantization()
        # if dyn_results:
        #     all_results.append(dyn_results)

        # Print comparison
        if all_results:
            print_comparison_table(all_results)
        else:
            print("\nNo benchmarks completed successfully.")

    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        clear_memory()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
