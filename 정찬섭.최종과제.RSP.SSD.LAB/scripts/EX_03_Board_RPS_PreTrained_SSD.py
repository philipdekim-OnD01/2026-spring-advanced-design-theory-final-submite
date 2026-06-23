# 모듈 로딩
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2 

# TFLite 모델 로딩
modelPath = 'RPS_PreTrained_SSD.tflite'
interpreter = tflite.Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
input_dtype = input_details[0]['dtype']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))
#print(input_details)
#print(output_details)

classList = '_ Scissors Rock Paper'.split()
colorList = [(),(255,0,0),(0,255,0),(0,0,255)]
IMG_SIZE = 320
threshold = 0.8

def processImage(frame):
    # frame 크기 저장 -> BB 표시할 때 사용 
    frameH, frameW, _ = frame.shape

    # BGR을 RGB로 변경
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # 모델의 입력 형태로 수정: (1,320,320,3)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img = np.expand_dims(img, 0)

    # -1 ~ 1 사이 값으로 변경
    img = img * (2/255) - 1

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기
    bboxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classIndexes = interpreter.get_tensor(output_details[3]['index'])[0].astype(int)
    classScores = interpreter.get_tensor(output_details[0]['index'])[0]

    for bbox, c, cs in zip(bboxes, classIndexes, classScores):
        # score가 threshold 이하이면 skip
        if cs <= threshold: continue

        # score를 100 분율로 환산
        classConfidence = round(cs*100 )
        
        # BB 표시
        classIndex = c + 1 # shift 필요: 0 1 2 -> 1 2 3
        classLabel = classList[classIndex]
        classColor = colorList[classIndex]
        displayText = f'{classLabel}: {classConfidence}%'.upper()
        ymin, xmin, ymax, xmax = bbox
        xmin, xmax, ymin, ymax = int(xmin*frameW), int(xmax*frameW), int(ymin*frameH), int(ymax*frameH)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
        cv2.putText(frame, displayText, (xmin, ymin-7), 
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=classColor, thickness=2)         

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

startTime = time.time()
while(cap.isOpened()):
    ret,frame=cap.read() # 사진 찍기 -> (240,320,3)
    if not ret: break

    # 이미지 처리
    processImage(frame)

    # FPS 표시
    curTime = time.time()
    fps = 1/(curTime - startTime)
    startTime = curTime
    cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)

    # 이미지 출력
    cv2.imshow('cam',frame)

     # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if  key == ord('q'): break

cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기