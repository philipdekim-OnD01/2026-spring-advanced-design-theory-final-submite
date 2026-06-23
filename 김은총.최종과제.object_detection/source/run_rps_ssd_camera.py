import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "RPS_PreTrained_SSD.tflite"

CLASS_LIST = "_ Scissors Rock Paper".split()
COLOR_LIST = [(), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
THRESHOLD = 0.8


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="TFLite model path. Defaults to the pretrained SSD camera model.",
    )
    return parser.parse_args()


def resolve_model_path(model_arg):
    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    return model_path


def preprocess(frame, input_shape, input_dtype, model_path):
    input_h = int(input_shape[1])
    input_w = int(input_shape[2])
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = np.expand_dims(img, 0)
    if model_path.name == "RPS_PreTrained_SSD.tflite":
        img = img * (2 / 255) - 1
    else:
        img = img / 255.0
    return img.astype(input_dtype)


def draw_tflite_ssd_detections(frame, boxes, class_indexes, scores):
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


def draw_ssd_lite_raw_detections(frame, predictions, max_boxes=3):
    frame_h, frame_w, _ = frame.shape
    class_logits = predictions[:, 4:7]
    scores = sigmoid(np.max(class_logits, axis=1))
    class_indexes = np.argmax(class_logits, axis=1) + 1

    for row_index in np.argsort(-scores)[:max_boxes]:
        score = float(scores[row_index])
        if score <= THRESHOLD:
            continue

        class_index = int(class_indexes[row_index])
        if class_index < 1 or class_index >= len(CLASS_LIST):
            continue

        xmin, ymin, xmax, ymax = predictions[row_index, :4]
        xmin = int(np.clip(xmin, 0, 1) * frame_w)
        xmax = int(np.clip(xmax, 0, 1) * frame_w)
        ymin = int(np.clip(ymin, 0, 1) * frame_h)
        ymax = int(np.clip(ymax, 0, 1) * frame_h)
        if xmax <= xmin or ymax <= ymin:
            continue

        label = CLASS_LIST[class_index]
        color = COLOR_LIST[class_index]
        confidence = round(score * 100)

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
    args = parse_args()
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    print("model:", model_path)
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

        img = preprocess(frame, input_shape, input_dtype, model_path)
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        if len(output_details) == 1:
            predictions = interpreter.get_tensor(output_details[0]["index"])[0]
            draw_ssd_lite_raw_detections(frame, predictions)
        else:
            scores = interpreter.get_tensor(output_details[0]["index"])[0]
            boxes = interpreter.get_tensor(output_details[1]["index"])[0]
            class_indexes = interpreter.get_tensor(output_details[3]["index"])[0].astype(int)
            draw_tflite_ssd_detections(frame, boxes, class_indexes, scores)

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
