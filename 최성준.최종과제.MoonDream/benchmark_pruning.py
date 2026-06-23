"""
Model Pruning Benchmark for Moondream
Tests various pruning strategies to reduce model size while maintaining quality
"""

import torch
import torch.nn.utils.prune as prune
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

# Pruning configurations to test
PRUNING_AMOUNTS = [0.1, 0.2, 0.3, 0.4, 0.5]  # 10%, 20%, 30%, 40%, 50%


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


def count_zero_parameters(model):
    """Count number of pruned (zero) parameters"""
    total_zeros = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight'):
                weight = module.weight
                total_params += weight.numel()
                total_zeros += (weight == 0).sum().item()

    return total_zeros, total_params


def get_model_size_mb(model):
    """Calculate actual model size in memory"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2


def get_sparsity(model):
    """Calculate global sparsity (percentage of zero weights)"""
    zeros, total = count_zero_parameters(model)
    if total == 0:
        return 0.0
    return 100.0 * zeros / total


def clear_memory():
    """Clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def apply_pruning(model, amount, method='l1'):
    """
    Apply pruning to model

    Args:
        model: The model to prune
        amount: Fraction of weights to prune (0.0 to 1.0)
        method: Pruning method ('l1', 'random', 'l2')
    """
    print(f"\nApplying {method.upper()} pruning with {amount*100:.0f}% sparsity...")

    pruned_modules = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if method == 'l1':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'random':
                prune.random_unstructured(module, name='weight', amount=amount)
            elif method == 'l2':
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            pruned_modules += 1

    print(f"Pruned {pruned_modules} Linear modules")

    # Get actual sparsity
    actual_sparsity = get_sparsity(model)
    print(f"Achieved sparsity: {actual_sparsity:.2f}%")

    return model


def make_pruning_permanent(model):
    """Remove pruning reparameterization and make it permanent"""
    print("Making pruning permanent...")

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass  # Module may not have been pruned

    return model


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
    zeros, total_linear = count_zero_parameters(model)
    sparsity = get_sparsity(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {zeros:,} ({sparsity:.2f}% sparse)")

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
            try:
                encoded_image = model.encode_image(image)
                _ = model.answer_question(encoded_image, TEST_QUESTION, tokenizer)
            except Exception as e:
                print(f"Warmup failed: {str(e)}")
                return None
        clear_memory()

    # Benchmark runs
    print(f"Running {NUM_TEST_RUNS} benchmark iterations...")
    inference_times = []
    answers = []

    for i in range(NUM_TEST_RUNS):
        clear_memory()

        start_time = time.time()
        with torch.no_grad():
            try:
                encoded_image = model.encode_image(image)
                answer = model.answer_question(encoded_image, TEST_QUESTION, tokenizer)
            except Exception as e:
                print(f"Inference failed: {str(e)}")
                return None

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
    print(f"  {answers[0][:150]}...")

    results = {
        'config_name': config_name,
        'total_params': total_params,
        'zero_params': zeros,
        'sparsity': sparsity,
        'model_size_mb': model_size_mb,
        'cpu_memory_mb': cpu_mem_after,
        'gpu_memory_mb': gpu_mem_after,
        'avg_inference_time_ms': avg_time,
        'min_inference_time_ms': min_time,
        'max_inference_time_ms': max_time,
        'sample_output': answers[0]
    }

    return results


def test_baseline():
    """Test baseline model without pruning"""
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL (NO PRUNING)")
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
    results = benchmark_model(model, tokenizer, image, "Baseline (FP16, No Pruning)")

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def test_pruned_model(pruning_amount, method='l1'):
    """Test model with specific pruning amount"""
    print("\n" + "="*60)
    print(f"LOADING MODEL WITH {pruning_amount*100:.0f}% {method.upper()} PRUNING")
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

    # Apply pruning
    model = apply_pruning(model, pruning_amount, method=method)

    # Make pruning permanent
    model = make_pruning_permanent(model)

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    config_name = f"{method.upper()} Pruning ({pruning_amount*100:.0f}%)"
    results = benchmark_model(model, tokenizer, image, config_name)

    # Cleanup
    del model
    del tokenizer
    clear_memory()

    return results


def save_pruned_model(pruning_amount, method='l1', output_dir='./pruned_models'):
    """Save pruned model for deployment"""
    print(f"\n{'='*60}")
    print(f"SAVING PRUNED MODEL ({pruning_amount*100:.0f}% sparsity)")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Apply and finalize pruning
    model = apply_pruning(model, pruning_amount, method=method)
    model = make_pruning_permanent(model)

    # Save model
    save_path = os.path.join(output_dir, f'moondream_pruned_{int(pruning_amount*100)}pct')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Check file size
    total_size = 0
    for root, dirs, files in os.walk(save_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    print(f"Model saved to: {save_path}")
    print(f"Total size on disk: {total_size / (1024**2):.2f} MB")

    del model
    del tokenizer
    clear_memory()

    return save_path, total_size / (1024**2)


def print_comparison_table(all_results):
    """Print comparison table of all results"""
    print("\n" + "="*80)
    print("PRUNING COMPARISON SUMMARY")
    print("="*80)

    # Table header
    print(f"\n{'Configuration':<30} {'Sparsity':<12} {'Model Size':<15} {'GPU Mem':<12} {'Avg Time':<12}")
    print(f"{'':<30} {'(%)':<12} {'(MB)':<15} {'(MB)':<12} {'(ms)':<12}")
    print("-" * 90)

    baseline = all_results[0]

    for result in all_results:
        config = result['config_name']
        sparsity = result.get('sparsity', 0)
        model_size = result['model_size_mb']
        gpu_mem = result['gpu_memory_mb']
        avg_time = result['avg_inference_time_ms']

        # Calculate improvements
        size_ratio = (baseline['model_size_mb'] / model_size) if model_size > 0 else 0
        mem_ratio = (baseline['gpu_memory_mb'] / gpu_mem) if gpu_mem > 0 else 0
        speed_ratio = (result['avg_inference_time_ms'] / baseline['avg_inference_time_ms'])

        print(f"{config:<30} {sparsity:>8.1f}%   {model_size:>10.2f} MB   {gpu_mem:>8.2f} MB   {avg_time:>8.2f} ms")

        if config != baseline['config_name']:
            print(f"{'  vs Baseline':<30} {'':<12} {size_ratio:>10.2f}x     {mem_ratio:>8.2f}x     {speed_ratio:>8.2f}x")

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    if len(all_results) > 1:
        best_pruned = max([r for r in all_results[1:]],
                         key=lambda x: x.get('sparsity', 0) if x['gpu_memory_mb'] < baseline['gpu_memory_mb'] * 0.9 else 0)

        if best_pruned:
            size_reduction = baseline['model_size_mb'] / best_pruned['model_size_mb']
            mem_reduction = baseline['gpu_memory_mb'] / best_pruned['gpu_memory_mb']
            speed_impact = best_pruned['avg_inference_time_ms'] / baseline['avg_inference_time_ms']

            print(f"\nBest Configuration: {best_pruned['config_name']}")
            print(f"  • Sparsity: {best_pruned['sparsity']:.1f}%")
            print(f"  • Size reduction: {size_reduction:.2f}x")
            print(f"  • Memory reduction: {mem_reduction:.2f}x")
            print(f"  • Speed impact: {speed_impact:.2f}x")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• 10-20% pruning: Minimal quality loss, good for production")
    print("• 30% pruning: Balanced size/quality trade-off")
    print("• 40-50% pruning: Aggressive, may degrade quality significantly")
    print("\nNOTE: Always validate output quality on your specific use case!")
    print("Pruning affects different model architectures differently.")


def main():
    print("="*80)
    print("MOONDREAM MODEL PRUNING BENCHMARK")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\n" + "="*80)
    print("ABOUT PRUNING")
    print("="*80)
    print("Model pruning removes low-magnitude weights to reduce model size.")
    print("This benchmark tests L1 unstructured pruning at different sparsity levels.")
    print(f"\nPruning amounts to test: {[f'{int(p*100)}%' for p in PRUNING_AMOUNTS]}")

    all_results = []

    try:
        # Test baseline
        print("\n" + "="*80)
        print("STEP 1: BASELINE MODEL")
        print("="*80)
        baseline_results = test_baseline()
        if baseline_results:
            all_results.append(baseline_results)

        # Test pruned models
        print("\n" + "="*80)
        print("STEP 2: PRUNED MODELS")
        print("="*80)

        for amount in PRUNING_AMOUNTS:
            pruned_results = test_pruned_model(amount, method='l1')
            if pruned_results:
                all_results.append(pruned_results)

        # Print comparison
        if all_results:
            print_comparison_table(all_results)
        else:
            print("\nNo benchmarks completed successfully.")

        # Ask if user wants to save best model
        if len(all_results) > 1:
            print("\n" + "="*80)
            print("OPTIONAL: SAVE PRUNED MODEL")
            print("="*80)
            print("To save a pruned model for deployment, run:")
            print(f"  python {__file__} --save --prune-amount 0.3")
            print("\nOr uncomment the save function in the script.")

    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        clear_memory()

    print("\n" + "="*80)
    print("PRUNING BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys

    # Simple CLI
    if "--save" in sys.argv:
        prune_amount = 0.3  # Default
        if "--prune-amount" in sys.argv:
            idx = sys.argv.index("--prune-amount")
            prune_amount = float(sys.argv[idx + 1])

        save_path, size = save_pruned_model(prune_amount)
        print(f"\nModel saved successfully!")
        print(f"Location: {save_path}")
        print(f"Size: {size:.2f} MB")
    else:
        main()
