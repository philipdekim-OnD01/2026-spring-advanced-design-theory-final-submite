"""
Rock-Paper-Scissors 2-player real-time judge
- Left: 1P / Right: 2P
- Q: quit
"""

import cv2
import numpy as np

# TFLite 자동 감지
def _load_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter

Interpreter = _load_interpreter_class()

MODEL_PATH     = "RPS_finetuned_rock.tflite"
IMG_SIZE       = 64
CLASS_NAMES    = ["Scissors", "Rock", "Paper"]
CONF_THRESHOLD = 0.6  # 60% 미만이면 "?" 표시

WIN_TABLE = {
    (0, 0): "Draw",   (0, 1): "2P Win", (0, 2): "1P Win",
    (1, 0): "1P Win", (1, 1): "Draw",   (1, 2): "2P Win",
    (2, 0): "2P Win", (2, 1): "1P Win", (2, 2): "Draw",
}
RESULT_COLOR = {
    "1P Win": (0, 200, 0),
    "2P Win": (0, 100, 255),
    "Draw":   (200, 200, 0),
    "?":      (100, 100, 100),
}

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp  = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]


def predict(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB (필수 — 없으면 색 반전으로 인식 불가)
    img = img.astype(np.float32)
    interpreter.set_tensor(inp["index"], np.expand_dims(img, 0))
    interpreter.invoke()
    probs = interpreter.get_tensor(outp["index"])[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def draw_text(frame, text, pos, size=1.0, color=(255,255,255), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, size, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    for _ in range(10):
        cap.read()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN_NAME = "RPS - 1P(left) vs 2P(right)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 640, 480)

    print("=== Rock-Paper-Scissors Real-time Judge ===")
    print("Q: quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        mid = w // 2

        pad    = 20
        p1_roi = frame[pad:h-pad, pad:mid-pad]
        p2_roi = frame[pad:h-pad, mid+pad:w-pad]

        c1, conf1 = predict(p1_roi)
        c2, conf2 = predict(p2_roi)

        # confidence 낮으면 결과 무효 처리
        if conf1 < CONF_THRESHOLD or conf2 < CONF_THRESHOLD:
            result       = "?"
            result_color = RESULT_COLOR["?"]
        else:
            result       = WIN_TABLE[(c1, c2)]
            result_color = RESULT_COLOR[result]

        # ROI 박스 & 구분선
        cv2.rectangle(frame, (pad, pad), (mid-pad, h-pad), (255, 100, 0), 2)
        cv2.rectangle(frame, (mid+pad, pad), (w-pad, h-pad), (0, 100, 255), 2)
        draw_text(frame, "1P", (pad+8, 36), size=1.0, color=(255,150,50), bg=False)
        draw_text(frame, "2P", (mid+pad+8, 36), size=1.0, color=(50,150,255), bg=False)
        cv2.line(frame, (mid, 0), (mid, h), (200, 200, 200), 1)

        # 클래스 + confidence 표시
        label1 = f"1P: {CLASS_NAMES[c1]} ({conf1*100:.0f}%)"
        label2 = f"2P: {CLASS_NAMES[c2]} ({conf2*100:.0f}%)"
        draw_text(frame, label1, (pad+8, h//2 - 20), size=0.9, color=(255, 200, 100))
        draw_text(frame, label2, (mid+pad+8, h//2 - 20), size=0.9, color=(100, 200, 255))

        # 결과
        draw_text(frame, result,
                  (w//2 - 60, h//2 + 60), size=2.0, color=result_color, thickness=4)

        cv2.imshow(WIN_NAME, frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
