import cv2
import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("tflite_runtime이 설치되지 않았습니다.")
    exit()

# 1. 모델 설정
model_path = 'best_int8.tflite'
print(f"[{model_path}] 모델 로드 중...")

try:
    # CPU 코어 4개를 모두 사용하도록 설정 (속도 향상)
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    print("✅ 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# COCO 80 전체 클래스 매핑
CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 2. 웹캠 초기화
cap = cv2.VideoCapture(0)
# 라즈베리파이 부하를 줄이기 위해 카메라 자체 해상도를 낮춤
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("객체 인식을 시작합니다. (종료: 'q' 키)")
prev_time = 0

# 신뢰도(Confidence) 기준점
CONF_THRESHOLD = 0.4 
NMS_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # 모델 입력 크기 (YOLOv8 기본값: 640x640)
    input_w, input_h = 640, 640
    frame_h, frame_w = frame.shape[:2]

    # --- [전처리] ---
    img = cv2.resize(frame, (input_w, input_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

    # --- [추론] ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # --- [후처리] ---
    predictions = np.squeeze(output_data)
    
    # 배열 형태 자동 교정 (8400, 84)로 무조건 맞춤
    if len(predictions.shape) == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T 
        
    boxes = []
    scores = []
    class_ids = []

    # 파이썬 반복문을 Numpy 벡터 연산으로 한방에 압축 (속도 향상)
    all_scores = np.max(predictions[:, 4:], axis=1)
    all_class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # 확률이 40%가 넘는 똑똑한 박스들의 번호(인덱스)만 쏙 골라냄
    valid_indices = np.where(all_scores > CONF_THRESHOLD)[0]

    # 유효한(통과한) 박스 몇 개만 반복해서 좌표 계산
    for i in valid_indices:
        row = predictions[i]
        score = all_scores[i]
        class_id = all_class_ids[i]

        cx, cy, w, h = row[0:4]
        
        # (안전장치) 모델이 0~1 사이로 정규화된 값을 뱉을 경우 640 스케일로 복원
        if w <= 1.0 and h <= 1.0:
            cx *= input_w
            cy *= input_h
            w *= input_w
            h *= input_h
        
        # 원본 카메라 비율에 맞게 좌표 스케일링
        x_scale = frame_w / input_w
        y_scale = frame_h / input_h
        
        cx = cx * x_scale
        cy = cy * y_scale
        w = w * x_scale
        h = h * y_scale
        
        # 좌상단 좌표(x, y)로 변환
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        
        boxes.append([x, y, int(w), int(h)])
        scores.append(float(score))
        class_ids.append(class_id)

    # 겹침 방지 (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    # --- [화면 출력] ---
    detected_items = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = CLASSES.get(class_ids[i], f'Class {class_ids[i]}')
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f'{label}: {scores[i]:.2f}'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected_items.append(f"{label}({scores[i]:.2f})")

    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if detected_items:
        print(f"👀 인식됨: {', '.join(detected_items)}")

    # 화면 띄우기
    cv2.imshow('YOLOv8 TFLite', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
