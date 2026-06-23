import cv2
import time
import torch
from PIL import Image
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
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model.eval()

print("Model loaded.")

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

last_inference = 0
inference_interval = 2.0  # seconds

current_answer = "Waiting..."

while True:

    ret, frame = cap.read()

    if not ret:
        break

    now = time.time()

    # Perform VLM inference every few seconds
    if now - last_inference > inference_interval:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        try:
            with torch.no_grad():

                encoded_image = model.encode_image(pil_image)

                answer = model.answer_question(
                    encoded_image,
                    "What is the main object in this image?",
                    tokenizer
                )

                current_answer = answer

        except Exception as e:
            current_answer = f"Error: {str(e)}"

        last_inference = now

    # Draw prediction
    cv2.putText(
        frame,
        current_answer[:80],
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Moondream Raspberry Pi Demo", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()