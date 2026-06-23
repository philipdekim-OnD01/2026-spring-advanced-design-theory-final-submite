#!/usr/bin/env python3
"""
Live Raspberry Pi camera inference for an RPS TFLite model + guarded local tiny LLM explanation.

v4 changes:
  - Compact overlay: small top status bar + small bottom explanation bar.
  - Automatic judgment every N seconds, default 7 seconds.
  - SPACE/touch still works as an optional immediate trigger.

Usage:
  DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority python3 rpi_camera_infer_tflite_llm_v4_auto7_compact.py \
      RPS_PreTrained_DenseNet_Augmentation_QAT.tflite \
      --labels rock,scissors,paper \
      --ollama-model qwen2.5:1.5b

Keys:
  SPACE or touchscreen/mouse click : optional immediate LLM explanation
  q/ESC : quit
"""

import argparse
from collections import Counter, deque
import random
import textwrap
import threading
import time
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import requests

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


RPS_MOVES = ["rock", "paper", "scissors"]
MOVE_KO = {"rock": "바위", "paper": "보", "scissors": "가위"}
RESULT_KO = {"win": "사용자 승리", "lose": "AI 승리", "draw": "무승부", "unknown": "결과 알 수 없음"}


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Crop the center square region to avoid distorting the hand shape."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def prepare_input(frame_bgr: np.ndarray, input_details) -> np.ndarray:
    """Convert OpenCV BGR frame to model input tensor."""
    _, input_h, input_w, _ = input_details[0]["shape"]

    crop_bgr = center_square_crop(frame_bgr)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (int(input_w), int(input_h)), interpolation=cv2.INTER_AREA)
    x = rgb[np.newaxis, ...].astype(np.float32)

    input_dtype = input_details[0]["dtype"]
    if input_dtype != np.float32:
        scale, zero_point = input_details[0]["quantization"]
        if scale == 0:
            raise ValueError("Invalid input quantization scale: 0")
        x = x / scale + zero_point
        x = np.clip(x, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        x = x.astype(input_dtype)

    return x


def to_probabilities(y: np.ndarray) -> np.ndarray:
    """Return a probability-like vector whether model output is probs or logits."""
    y = np.asarray(y).astype(np.float32).reshape(-1)

    if np.all(y >= 0.0) and np.all(y <= 1.0) and 0.90 <= float(np.sum(y)) <= 1.10:
        return y

    y = y - np.max(y)
    exp_y = np.exp(y)
    return exp_y / np.sum(exp_y)


def run_inference(interpreter, input_details, output_details, frame_bgr: np.ndarray) -> np.ndarray:
    x = prepare_input(frame_bgr, input_details)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])[0]

    output_dtype = output_details[0]["dtype"]
    if output_dtype != np.float32:
        scale, zero_point = output_details[0]["quantization"]
        y = scale * (y.astype(np.float32) - zero_point)

    return to_probabilities(y)


def decide_result(user_move: str, ai_move: str) -> str:
    """Return win/lose/draw from the user's perspective."""
    user_move = user_move.lower().strip()
    ai_move = ai_move.lower().strip()

    if user_move == ai_move:
        return "draw"

    win_rule = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }

    if user_move not in win_rule or ai_move not in win_rule:
        return "unknown"

    return "win" if win_rule[user_move] == ai_move else "lose"


def fallback_explanation(user_move: str, ai_move: str, result: str, confidence: float) -> str:
    user_ko = MOVE_KO.get(user_move, user_move)
    ai_ko = MOVE_KO.get(ai_move, ai_move)
    result_ko = RESULT_KO.get(result, result)
    conf_msg = "high / reliable" if confidence >= 0.75 else "low / try better lighting"

    if result == "draw":
        reason_en = f"Both selected {user_move}."
    elif result == "win":
        reason_en = f"{user_move} beats {ai_move}."
    elif result == "lose":
        reason_en = f"{ai_move} beats {user_move}."
    else:
        reason_en = "Unknown input."

    # English is used on the OpenCV overlay because cv2.putText does not reliably render Korean.
    # Korean equivalent is printed to the terminal by the LLM when available.
    return (
        f"User: {user_move}({user_ko}), AI: {ai_move}({ai_ko}) -> {result_ko}. "
        f"{reason_en} Confidence {confidence:.2f} is {conf_msg}."
    )


def response_looks_bad(text: str) -> bool:
    lowered = text.lower()
    bad_terms = [
        "go", "바둑", "서울", "can't assist", "cannot assist", "보기 어렵",
        "impossible", "stone", "board", "grid", "오목", "체스", "장기"
    ]
    return (not text.strip()) or any(term in lowered for term in bad_terms) or len(text) > 320


def response_consistent_with_result(text: str, result: str) -> bool:
    """Check whether the LLM text agrees with the Python-computed winner."""
    t = text.lower()

    user_win_terms = ["사용자 승리", "사용자가 이", "당신이 이", "user win", "user won", "you win", "you won"]
    ai_win_terms = ["ai 승리", "ai가 이", "제가 이", "내가 이", "assistant win", "ai win", "ai won", "i win", "i won"]
    draw_terms = ["무승부", "비겼", "동점", "draw", "tie"]

    has_user_win = any(x in t for x in user_win_terms)
    has_ai_win = any(x in t for x in ai_win_terms)
    has_draw = any(x in t for x in draw_terms)

    if result == "win":
        return has_user_win and not has_ai_win and not has_draw
    if result == "lose":
        return has_ai_win and not has_user_win and not has_draw
    if result == "draw":
        return has_draw and not has_user_win and not has_ai_win
    return False


def build_prompt(user_move: str, ai_move: str, result: str, confidence: float, fps: float) -> str:
    user_ko = MOVE_KO.get(user_move, user_move)
    ai_ko = MOVE_KO.get(ai_move, ai_move)
    result_ko = RESULT_KO.get(result, result)
    return f"""
You are a fixed-output assistant for a Raspberry Pi rock-paper-scissors demo.
Do not reinterpret the game. The result below was already computed by Python.

Rules:
- rock beats scissors
- scissors beats paper
- paper beats rock
- same move is draw

User move: {user_move} ({user_ko})
AI move: {ai_move} ({ai_ko})
Computed result from user's perspective: {result} ({result_ko})
Classifier confidence: {confidence:.2f}
Camera FPS: {fps:.1f}

Write exactly two short Korean sentences.
Sentence 1 must clearly state who won.
Sentence 2 must briefly mention confidence.
No markdown. No extra explanation.
""".strip()


def call_ollama(prompt: str, model: str, url: str, timeout_s: float) -> Tuple[str, Dict[str, float]]:
    """Call the local Ollama server and return response text plus simple metrics."""
    t0 = time.time()
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 60,
            },
        },
        timeout=timeout_s,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    text = data.get("response", "").strip()
    total_duration_s = data.get("total_duration", 0) / 1e9 if data.get("total_duration") else elapsed
    eval_count = float(data.get("eval_count", 0) or 0)
    eval_duration_s = data.get("eval_duration", 0) / 1e9 if data.get("eval_duration") else 0.0
    tok_s = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0

    metrics = {
        "latency_s": float(total_duration_s),
        "tokens_per_s": float(tok_s),
        "eval_count": float(eval_count),
    }
    return text, metrics


def get_stable_prediction(
    history: Deque[Tuple[float, str, float]],
    fallback_label: str,
    fallback_score: float,
    window_s: float,
) -> Tuple[str, float]:
    """Use a short majority vote window so the 7-second auto judgment is less flickery."""
    now = time.time()
    recent = [(label, score) for ts, label, score in history if now - ts <= window_s]
    if not recent:
        return fallback_label, fallback_score

    counts = Counter(label for label, _ in recent)
    stable_label = counts.most_common(1)[0][0]
    scores = [score for label, score in recent if label == stable_label]
    stable_score = float(np.mean(scores)) if scores else fallback_score
    return stable_label, stable_score


def draw_text_lines(frame: np.ndarray, lines: List[str], x: int, y: int, font_scale: float = 0.48, line_h: int = 20) -> None:
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        y += line_h


def draw_overlay(
    frame: np.ndarray,
    label: str,
    score: float,
    probs: np.ndarray,
    fps: float,
    labels,
    llm_state: Dict,
    auto_interval: float,
    next_auto_time: float,
) -> None:
    h, w = frame.shape[:2]

    # Compact top status bar. This replaces the large black box from v3.
    top_h = 72
    cv2.rectangle(frame, (0, 0), (w, top_h), (0, 0, 0), thickness=-1)

    status = llm_state.get("status", "ready")
    ai_move = llm_state.get("ai_move", "-")
    result = llm_state.get("result", "-")
    remain = max(0.0, next_auto_time - time.time()) if auto_interval > 0 else 0.0

    top = f"TFLite: {label} ({score:.2f}) | FPS {fps:.1f} | Auto {remain:.0f}s | LLM {status}"
    cv2.putText(frame, top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    prob_text = "  ".join(f"{name}:{prob:.2f}" for name, prob in zip(labels, probs))
    cv2.putText(frame, prob_text, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"AI: {ai_move} | Result: {result} | q/ESC quit", (10, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Small bottom explanation bar. It does not cover the hand/crop area much.
    bottom_h = 88
    y0 = h - bottom_h
    cv2.rectangle(frame, (0, y0), (w, h), (0, 0, 0), thickness=-1)

    explanation = llm_state.get("text", "Auto judgment runs every 7 seconds.")
    metrics = llm_state.get("metrics", {})
    metric_text = ""
    if metrics:
        metric_text = f" [lat {metrics.get('latency_s', 0):.1f}s, {metrics.get('tokens_per_s', 0):.1f} tok/s]"

    wrapped: List[str] = []
    for paragraph in (explanation + metric_text).splitlines():
        wrapped.extend(textwrap.wrap(paragraph, width=78) or [""])

    draw_text_lines(frame, wrapped[:3], 10, y0 + 22, font_scale=0.46, line_h=19)

    # Show the center crop box used for inference.
    side = min(h, w)
    crop_y0 = (h - side) // 2
    crop_x0 = (w - side) // 2
    cv2.rectangle(frame, (crop_x0, crop_y0), (crop_x0 + side, crop_y0 + side), (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera RPS inference with TFLite + local tiny LLM")
    parser.add_argument("model", help="Path to .tflite model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index, usually 0")
    parser.add_argument("--labels", default="rock,scissors,paper", help="Comma-separated class names in training label order")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--threads", type=int, default=4, help="TFLite interpreter threads")
    parser.add_argument("--ollama-model", default="qwen2.5:1.5b", help="Local Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate API URL")
    parser.add_argument("--llm-timeout", type=float, default=45.0, help="LLM request timeout in seconds")
    parser.add_argument("--auto-interval", type=float, default=7.0, help="Automatically ask LLM every N seconds. Use 0 to disable auto mode.")
    parser.add_argument("--stable-window", type=float, default=2.0, help="Seconds of recent predictions used for majority vote before LLM judgment")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM call and only show TFLite classifier")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",")]
    if len(labels) != 3:
        raise ValueError("--labels must contain exactly 3 labels, e.g. rock,scissors,paper")

    interpreter = Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded:", args.model)
    print("Input details:", input_details[0])
    print("Output details:", output_details[0])
    print("Labels:", labels)
    print("Ollama model:", args.ollama_model)
    print(f"Auto judgment interval: {args.auto_interval}s")
    print("SPACE/tap still triggers an immediate judgment. q/ESC quits.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Check camera connection, enable camera, or try --camera 1."
        )

    fps = 0.0
    last_time = time.time()
    prediction_history: Deque[Tuple[float, str, float]] = deque(maxlen=180)

    llm_state = {
        "status": "disabled" if args.no_llm else "ready",
        "text": "Auto judgment runs every 7 seconds. SPACE/tap = judge now." if not args.no_llm else "LLM is disabled.",
        "ai_move": "-",
        "result": "-",
        "metrics": {},
    }
    llm_lock = threading.Lock()

    request_llm_from_touch = {"flag": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not args.no_llm:
            request_llm_from_touch["flag"] = True

    window_name = "RPS TFLite + Tiny LLM Auto"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    def start_llm_worker(user_move: str, confidence: float, current_fps: float, source: str) -> None:
        normalized_user_move = user_move.lower().strip()
        ai_move = random.choice(RPS_MOVES)
        result = decide_result(normalized_user_move, ai_move)

        with llm_lock:
            llm_state.update({
                "status": "thinking",
                "text": f"Judging now... source={source}, user={normalized_user_move}",
                "ai_move": ai_move,
                "result": result,
                "metrics": {},
            })

        def worker():
            prompt = build_prompt(normalized_user_move, ai_move, result, confidence, current_fps)
            try:
                text, metrics = call_ollama(prompt, args.ollama_model, args.ollama_url, args.llm_timeout)
                terminal_text = text
                if response_looks_bad(text) or not response_consistent_with_result(text, result):
                    text = fallback_explanation(normalized_user_move, ai_move, result, confidence)
                    metrics["fallback_used"] = 1.0
                with llm_lock:
                    llm_state.update({
                        "status": "ready",
                        "text": text,
                        "metrics": metrics,
                    })
                print("\n=== Auto RPS judgment ===")
                print(f"source={source}, user={normalized_user_move}, ai={ai_move}, result={result}, confidence={confidence:.2f}")
                print("LLM raw:", terminal_text)
                print("UI text:", text)
                print("Metrics:", metrics)
            except Exception as e:
                safe = fallback_explanation(normalized_user_move, ai_move, result, confidence)
                with llm_lock:
                    llm_state.update({
                        "status": "ready",
                        "text": safe + f" LLM call failed: {e}",
                        "metrics": {},
                    })
                print("LLM call failed:", e)

        threading.Thread(target=worker, daemon=True).start()

    next_auto_time = time.time() + max(0.0, args.auto_interval)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera")
            break

        probs = run_inference(interpreter, input_details, output_details, frame)
        pred = int(np.argmax(probs))
        score = float(probs[pred])
        label = labels[pred] if pred < len(labels) else str(pred)
        prediction_history.append((time.time(), label, score))

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        with llm_lock:
            state_snapshot = dict(llm_state)

        draw_overlay(frame, label, score, probs, fps, labels, state_snapshot, args.auto_interval, next_auto_time)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        manual_requested = (key == ord(" ")) or request_llm_from_touch["flag"]
        if manual_requested:
            request_llm_from_touch["flag"] = False

        auto_requested = False
        if (not args.no_llm) and args.auto_interval > 0 and now >= next_auto_time:
            auto_requested = True

        if (manual_requested or auto_requested) and not args.no_llm:
            with llm_lock:
                busy = llm_state.get("status") == "thinking"

            if not busy:
                stable_label, stable_score = get_stable_prediction(
                    prediction_history, label, score, max(0.1, args.stable_window)
                )
                source = "manual" if manual_requested else "auto"
                start_llm_worker(stable_label, stable_score, fps, source)
                next_auto_time = now + args.auto_interval if args.auto_interval > 0 else now
            elif auto_requested:
                # If the model is still thinking, try again soon instead of queuing many requests.
                next_auto_time = now + 1.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
