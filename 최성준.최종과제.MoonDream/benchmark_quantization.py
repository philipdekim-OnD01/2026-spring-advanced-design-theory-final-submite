"""
Comprehensive Quantization Benchmark for Moondream Model
Tests memory usage, inference time, and quality before/after quantization
"""

import torch
import time
import psutil
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

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

    # Get model size estimation
    param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_size_mb = sum(b.nelement() * b.element_size() for b in model.buffers()) / 1024**2
    total_size_mb = param_size_mb + buffer_size_mb

    print(f"\nModel Size:")
    print(f"  Parameters: {param_size_mb:.2f} MB")
    print(f"  Buffers: {buffer_size_mb:.2f} MB")
    print(f"  Total: {total_size_mb:.2f} MB")

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
    print(f"  {answers[0]}")

    results = {
        'config_name': config_name,
        'total_params': total_params,
        'model_size_mb': total_size_mb,
        'cpu_memory_mb': cpu_mem_after,
        'gpu_memory_mb': gpu_mem_after,
        'avg_inference_time_ms': avg_time,
        'min_inference_time_ms': min_time,
        'max_inference_time_ms': max_time,
        'sample_output': answers[0]
    }

    return results


def test_baseline():
    """Test baseline model without quantization"""
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL (FP16)")
    print("="*60)

    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16  # FP16 for efficiency
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "Baseline (FP16)")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_8bit_quantization():
    """Test 8-bit quantization"""
    print("\n" + "="*60)
    print("LOADING 8-BIT QUANTIZED MODEL")
    print("="*60)

    clear_memory()

    # Check if CUDA is available for quantization
    if not torch.cuda.is_available():
        print("WARNING: Quantization requires CUDA. Skipping 8-bit test.")
        return None

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Try loading with device_map first, fallback if not supported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map={"": 0}  # Explicitly map to GPU 0
        )
    except (ValueError, NotImplementedError):
        # Fallback: load without device_map
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model = model.to("cuda")

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "8-bit Quantization")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_4bit_quantization():
    """Test 4-bit NF4 quantization"""
    print("\n" + "="*60)
    print("LOADING 4-BIT QUANTIZED MODEL (NF4)")
    print("="*60)

    clear_memory()

    # Check if CUDA is available for quantization
    if not torch.cuda.is_available():
        print("WARNING: Quantization requires CUDA. Skipping 4-bit NF4 test.")
        return None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Try loading with device_map first, fallback if not supported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map={"": 0}  # Explicitly map to GPU 0
        )
    except (ValueError, NotImplementedError):
        # Fallback: load without device_map
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model = model.to("cuda")

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "4-bit NF4 Quantization")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_4bit_fp4_quantization():
    """Test 4-bit FP4 quantization (alternative to NF4)"""
    print("\n" + "="*60)
    print("LOADING 4-BIT QUANTIZED MODEL (FP4)")
    print("="*60)

    clear_memory()

    # Check if CUDA is available for quantization
    if not torch.cuda.is_available():
        print("WARNING: Quantization requires CUDA. Skipping 4-bit FP4 test.")
        return None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="fp4",  # FP4 instead of NF4
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Try loading with device_map first, fallback if not supported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map={"": 0}  # Explicitly map to GPU 0
        )
    except (ValueError, NotImplementedError):
        # Fallback: load without device_map
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model = model.to("cuda")

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    results = benchmark_model(model, tokenizer, image, "4-bit FP4 Quantization")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def print_comparison_table(all_results):
    """Print comparison table of all results"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Table header
    print(f"\n{'Configuration':<25} {'Model Size':<15} {'GPU Memory':<15} {'Avg Time':<15}")
    print(f"{'':<25} {'(MB)':<15} {'(MB)':<15} {'(ms)':<15}")
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

        print(f"{config:<25} {model_size:>10.2f} MB   {gpu_mem:>10.2f} MB   {avg_time:>10.2f} ms")

        if config != baseline['config_name']:
            print(f"{'  vs Baseline':<25} {size_ratio:>10.2f}x     {mem_ratio:>10.2f}x     {speed_ratio:>10.2f}x")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• 4-bit NF4: Best memory savings (~4x), slight speed trade-off")
    print("• 4-bit FP4: Alternative to NF4, may have different accuracy characteristics")
    print("• 8-bit: Good balance between size and quality (~2x memory savings)")
    print("• Baseline FP16: Best quality, highest memory usage")
    print("\nFor deployment on resource-constrained devices, use 4-bit NF4.")
    print("For best quality with some memory savings, use 8-bit quantization.")


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

    all_results = []

    try:
        # Test baseline
        baseline_results = test_baseline()
        if baseline_results:
            all_results.append(baseline_results)

        # Test 8-bit
        results_8bit = test_8bit_quantization()
        if results_8bit:
            all_results.append(results_8bit)

        # Test 4-bit NF4
        results_4bit_nf4 = test_4bit_quantization()
        if results_4bit_nf4:
            all_results.append(results_4bit_nf4)

        # Test 4-bit FP4
        results_4bit_fp4 = test_4bit_fp4_quantization()
        if results_4bit_fp4:
            all_results.append(results_4bit_fp4)

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


if __name__ == "__main__":
    main()