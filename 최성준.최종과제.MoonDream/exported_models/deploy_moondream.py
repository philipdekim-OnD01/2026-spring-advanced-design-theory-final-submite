"""
Moondream Deployment Script
Load and use the exported model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Configuration
MODEL_PATH = "./moondream_fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the model for inference"""
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(DEVICE)

    model.eval()

    print(f"Model loaded on {DEVICE}")
    return model, tokenizer

def predict(model, tokenizer, image_path, question="Please describe in detail"):
    """Run inference on an image"""
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        encoded_image = model.encode_image(image)
        answer = model.answer_question(encoded_image, question, tokenizer)

    return answer

if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model()

    # Example usage
    image_path = "test_image.png"
    question = "What do you see in this image?"

    answer = predict(model, tokenizer, image_path, question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
