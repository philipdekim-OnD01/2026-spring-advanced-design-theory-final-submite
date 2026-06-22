from pathlib import Path
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "RPS_PreTrained_SSD.tflite"

CLASS_LIST = "_ Scissors Rock Paper".split()
COLOR_LIST = [(), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
IMG_SIZE = 320
THRESHOLD = 0.8


def preprocess(frame, input_dtype):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img = img * (2 / 255) - 1
    return img.astype(input_dtype)


def draw_detections(frame, boxes, class_indexes, scores):
    frame_h, frame_w, _ = frame.shape

    for bbox, class_index_raw, score in zip(boxes, class_indexes, scores):
        if score <= THRESHOLD:
            continue

        class_index = int(class_index_raw) + 1
        if class_index < 1 or class_index >= len(CLASS_LIST):
            continue

        label = CLASS_LIST[class_index]
        color = COLOR_LIST[class_index]
        confidence = round(float(score) * 100)

        ymin, xmin, ymax, xmax = bbox
        xmin = int(xmin * frame_w)
        xmax = int(xmax * frame_w)
        ymin = int(ymin * frame_h)
        ymax = int(ymax * frame_h)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
        cv2.putText(
            frame,
            f"{label}: {confidence}%".upper(),
            (xmin, max(20, ymin - 7)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            2,
        )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    print("model:", MODEL_PATH)
    print("input:", input_details)
    print("outputs:", output_details)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess(frame, input_dtype)
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]["index"])[0]
        boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        class_indexes = interpreter.get_tensor(output_details[3]["index"])[0].astype(int)

        draw_detections(frame, boxes, class_indexes, scores)

        now = time.time()
        fps = 1 / (now - start_time)
        start_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        cv2.imshow("RPS SSD Object Detection", frame)
        if cv2.waitKey(10) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
