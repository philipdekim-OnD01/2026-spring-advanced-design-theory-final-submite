# Moondream Optimization - Quick Start

## 🚀 TL;DR

```bash
# Run everything automatically
./run_optimization_pipeline.sh
```

That's it! The pipeline will:
1. Test baseline FP16 model
2. Test model pruning (10%, 20%, 30%, 40%, 50%)
3. Export optimized models
4. Generate comprehensive report

**Note**: BitsAndBytes quantization is NOT included (incompatible with Moondream)

---

## ⚡ Quick Commands

### Run Complete Pipeline
```bash
./run_optimization_pipeline.sh
```

### Run Individual Tests
```bash
# Baseline only (fastest)
./run_optimization_pipeline.sh --baseline

# Pruning only
./run_optimization_pipeline.sh --pruning

# Export only
./run_optimization_pipeline.sh --export

# Quick test (baseline + export)
./run_optimization_pipeline.sh --quick
```

### Run Python Scripts Directly
```bash
# Baseline FP16 benchmark
python benchmark_quantization_pytorch.py

# Pruning benchmark
python benchmark_pruning.py

# Export model
python export_onnx.py

# Optional: BitsAndBytes test (educational only - will fail)
# python benchmark_quantization.py
```

---

## 📊 Expected Results

| Method | Size | GPU VRAM | Speed | Quality | Works? |
|--------|------|----------|-------|---------|--------|
| FP16 | 3.68 GB | 4.2 GB | 2.4s | 100% | ✅ |
| 8-bit | ~2.0 GB | ~2.0 GB | N/A | N/A | ❌ |
| 4-bit | ~1.0 GB | ~1.0 GB | N/A | N/A | ❌ |
| Prune 30% | 2.6 GB | 3.0 GB | 2.4s | 90% | ✅ |

**Recommendation**: Use FP16 or Pruned 20-30%

---

## 🎯 Use Cases

### I want the smallest model that works
```bash
python benchmark_pruning.py --save --prune-amount 0.3
# Output: ./pruned_models/moondream_pruned_30pct
```

### I want the best quality
```bash
# Use FP16 (already optimal)
python export_onnx.py
# Output: ./exported_models/moondream_fp16
```

### I want to test everything
```bash
./run_optimization_pipeline.sh
# Output: ./optimization_results/summary_report_*.md
```

### I want to deploy to production
```bash
python export_onnx.py
cd exported_models
python deploy_moondream.py  # Test it
./build_docker.sh           # Deploy it
```

---

## 📁 Output Files

After running the pipeline:

```
RP2-RPS-QAT-Lab/
├── optimization_results/
│   ├── baseline_*.log           # Baseline test results
│   ├── pruning_*.log            # Pruning test results
│   └── summary_report_*.md      # Final summary
│
├── exported_models/
│   ├── moondream_fp16/          # Optimized model
│   ├── deploy_moondream.py      # Deployment script
│   └── README.md                # Deployment docs
│
└── pruned_models/               # (if saved)
    └── moondream_pruned_*pct/
```

---

## 🔍 Check Results

```bash
# View summary report
cat optimization_results/summary_report_*.md

# View detailed logs
less optimization_results/baseline_*.log
less optimization_results/pruning_*.log

# Test exported model
cd exported_models
python deploy_moondream.py
```

---

## ❓ FAQ

**Q: Which method should I use?**
A: FP16 for best quality, Pruned 30% for smallest size

**Q: Why does 8-bit/4-bit fail?**
A: Moondream uses custom layers incompatible with bitsandbytes

**Q: How long does the pipeline take?**
A: ~15-30 minutes depending on your GPU

**Q: Do I need to run everything?**
A: No! Use `--baseline` for quick test, `--all` for complete analysis

**Q: Can I run on CPU?**
A: Yes, but 30-50x slower. GPU recommended.

---

## 💡 Pro Tips

1. **Run baseline first** to verify setup:
   ```bash
   ./run_optimization_pipeline.sh --baseline
   ```

2. **Save your best model**:
   ```bash
   python benchmark_pruning.py --save --prune-amount 0.2
   ```

3. **Test quality** on your own images:
   ```bash
   cd exported_models
   # Edit deploy_moondream.py to use your images
   python deploy_moondream.py
   ```

4. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## 📚 Full Documentation

- **Complete Guide**: `OPTIMIZATION_GUIDE.md`
- **Test Results**: `QUANTIZATION_RESULTS.md`
- **Theory**: `QUANTIZATION_BENCHMARK.md`

---

## 🆘 Troubleshooting

### CUDA Out of Memory
```bash
# Use pruned model or run on CPU
python benchmark_pruning.py --save --prune-amount 0.4
```

### Pipeline fails
```bash
# Check logs
tail -f optimization_results/pipeline_*.log
```

### Slow inference
```python
# Use FP16 and GPU
model = model.half().to("cuda")
```

---

**Need help?** Check `OPTIMIZATION_GUIDE.md` for detailed troubleshooting.

**Ready to deploy?** See `exported_models/README.md` for deployment options.
