# Moondream Exported Model

## Model Information
- **Original Model**: vikhyatk/moondream2
- **Precision**: FP16 (Half Precision)
- **Size**: 3680.65 MB
- **Format**: PyTorch SafeTensors

## Requirements

```bash
pip install torch transformers pillow
```

## Usage

### Option 1: Using the deployment script

```python
python deploy_moondream.py
```

### Option 2: Direct usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./moondream_fp16",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "./moondream_fp16",
    trust_remote_code=True
)

# Inference
image = Image.open("image.png").convert("RGB")
encoded_image = model.encode_image(image)
answer = model.answer_question(encoded_image, "Describe this image", tokenizer)
print(answer)
```

## System Requirements

- **GPU**: NVIDIA GPU with 5+ GB VRAM (for CUDA)
- **CPU**: Works on CPU but slower (30-50x)
- **RAM**: 4+ GB system RAM

## Performance

- **Inference Time**: ~2-3 seconds per image (GPU)
- **GPU Memory**: ~4.2 GB peak usage
- **Batch Size**: 1 (single image at a time)

## Notes

⚠️ **ONNX Export Not Supported**

This model uses custom code (`trust_remote_code=True`) which prevents
standard ONNX export. Use this PyTorch format for deployment instead.

For production deployment, consider:
1. Using this PyTorch FP16 model (recommended)
2. TorchScript compilation (if compatible)
3. Model pruning for further size reduction
4. Quantization-aware training for INT8 support

## Troubleshooting

### CUDA Out of Memory
- Reduce image resolution before processing
- Use CPU instead: `model.to("cpu")`
- Close other GPU applications

### Slow Inference
- Ensure you're using GPU: `model.to("cuda")`
- Use FP16: `torch_dtype=torch.float16`
- Enable torch.compile (PyTorch 2.0+): `model = torch.compile(model)`

## License

Follows the original Moondream model license.
