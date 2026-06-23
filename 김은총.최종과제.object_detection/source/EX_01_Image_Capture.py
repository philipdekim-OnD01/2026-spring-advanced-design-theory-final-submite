# 모듈 로딩
import cv2 

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

capture_dir = './' # 경로 변경 가능
cnt = 1 # 사진 번호

while(cap.isOpened()):
    ret,frame=cap.read() # 사진 찍기 -> (480,640,3)
    if not ret: break

    # 이미지 출력
    cv2.imshow('cam',frame)

     # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10) 
    if key == ord('q'): break
    if key == ord('s'): # 이미지 저장
        cv2.imwrite(f'{capture_dir}/img_{cnt:04d}.jpg', frame)
        cnt +=1

cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기