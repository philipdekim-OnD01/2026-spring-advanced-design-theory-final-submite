from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "vikhyatk/moondream2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

model.eval()

print("Model loaded.")

# image path
image_path = "/SSD1/sjchoi/semiai/RP2-RPS-QAT-Lab/RPS_Dataset/1/rock_1.png"
image_path = "/SSD1/sjchoi/semiai/RP2-RPS-QAT-Lab/cap/20260615_155214.png"
# image_path = "/SSD1/sjchoi/semiai/RP2-RPS-QAT-Lab/cap/20260615_155234.png"

image = Image.open(image_path).convert("RGB")

with torch.no_grad():

    encoded_image = model.encode_image(image)

    answer = model.answer_question(
        encoded_image,
        "Please describe in detail",
        tokenizer
    )

print("Answer:")
print(answer)