import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2 

# TFLite 모델 로딩 (현재 폴더의 모델 설정)
modelPath = 'RPS_PreTrained_SSD.tflite'
interpreter = tflite.Interpreter(model_path = modelPath) 
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details()  
output_details = interpreter.get_output_details() 
input_dtype = input_details[0]['dtype']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))

classList = '_ Scissors Rock Paper'.split()
threshold = 0.8

# [추가] 가위바위보 상성 판정 알고리즘 (1: p1 승, -1: p2 승, 0: 무승부)
def judge_rps(p1_type, p2_type):
    if p1_type == p2_type:
        return 0
    # 2(Rock) vs 1(Scissors), 3(Paper) vs 2(Rock), 1(Scissors) vs 3(Paper)
    if (p1_type == 2 and p2_type == 1) or (p1_type == 3 and p2_type == 2) or (p1_type == 1 and p2_type == 3):
        return 1
    return -1

# [추가] 검출된 손들의 목록을 비교하여 승자의 인덱스 추출
def get_game_winners(valid_detections):
    if len(valid_detections) != 2:
        return []  # 화면에 손이 정확히 2개 있을 때만 승패를 가림
        
    p1 = valid_detections[0]
    p2 = valid_detections[1]
    result = judge_rps(p1['class_index'], p2['class_index'])
    
    if result == 1:
        return [0]  # 첫 번째 손 승리
    elif result == -1:
        return [1]  # 두 번째 손 승리
    else:
        return []   # 무승부

# [수정] 여러 손을 모두 모아서 승패를 비교한 후 화면에 그리는 후처리 함수
def processImage(frame, boxes, class_indexes, classScores):
    frameH, frameW, _ = frame.shape
    valid_detections = []

    # 1. Threshold를 넘는 유효한 손들을 리스트에 먼저 바인딩
    for bbox, c, cs in zip(boxes, class_indexes, classScores):
        if cs <= threshold: 
            continue

        class_index = int(c) + 1  # shift 필요: 0 1 2 -> 1 2 3
        if class_index < 1 or class_index >= len(classList):
            continue

        valid_detections.append({
            'bbox': bbox,
            'class_index': class_index,
            'score': cs
        })

    # 2. 승패 판정 레이어 구동
    winner_indices = get_game_winners(valid_detections)

    # 3. 결과를 바탕으로 각기 다른 색상과 WINNER/LOSE 텍스트 렌더링
    for idx, det in enumerate(valid_detections):
        ymin, xmin, ymax, xmax = det['bbox']
        xmin, xmax, ymin, ymax = int(xmin*frameW), int(xmax*frameW), int(ymin*frameH), int(ymax*frameH)
        
        label = classList[det['class_index']]
        confidence = round(float(det['score']) * 100)

        # 손이 2개 펼쳐졌을 때만 게임 스코어 UI 분기 적용
        if len(valid_detections) == 2:
            if len(winner_indices) == 0:
                color = (255, 0, 0)  # Blue (무승부)
                displayText = f'[DRAW] {label}: {confidence}%'
            elif idx in winner_indices:
                color = (0, 255, 0)  # Green (승리)
                displayText = f'[WINNER] {label}: {confidence}%'
            else:
                color = (0, 0, 255)  # Red (패배)
                displayText = f'[LOSE] {label}: {confidence}%'
        else:
            # 손이 1개이거나 3개 이상일 때는 기본 파란색 상자로 일반 출력
            color = (255, 0, 0)
            displayText = f'{label}: {confidence}%'

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=3)
        cv2.putText(frame, displayText.upper(), (xmin, ymin-10), 
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=color, thickness=2)

# 카메라 구동 메인 프레임 루프
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

startTime = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: break

    # 전처리 및 모델 인프라 추론 구동
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE:=320, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img = img * (2/255) - 1

    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    interpreter.invoke()

    # 결과 텐서 추출
    classScores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    class_indexes = interpreter.get_tensor(output_details[3]['index'])[0]

    # [수정 반영한 후처리 함수 호출]
    processImage(frame, boxes, class_indexes, classScores)

    # 실시간 FPS 렌더링
    endTime = time.time()
    fps = 1 / (endTime - startTime)
    startTime = endTime
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow('RPS Object Detection Game', frame)
    if cv2.waitKey(10) == ord('q'): break

cap.release()
cv2.destroyAllWindows()