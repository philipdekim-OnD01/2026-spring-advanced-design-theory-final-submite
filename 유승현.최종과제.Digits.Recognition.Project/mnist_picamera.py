import os
import cv2
import numpy as np
import tensorflow as tf
import time

# Model load
MODEL_PATH = "digits_model_best.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} 파일이 현재 폴더에 없습니다.")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded:", MODEL_PATH)
print("Model input shape:", model.input_shape)

frame_width = 640
frame_height = 480
frame_rate = 16

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("USB 카메라를 열 수 없습니다.")
    print("카메라 번호를 0에서 1로 바꿔보세요.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

time.sleep(0.5)

print("USB camera started.")
print("Press q to quit.")


# One digit preprocessing

def preprocess_digit_for_mnist(img_roi):
    """
    img_roi:
        threshold image.
        black background + white digit 형태여야 함.

    return:
        28x28 float32 image, range [0, 1]
    """

    if img_roi is None or img_roi.size == 0:
        return None

    
    _, roi = cv2.threshold(img_roi, 127, 255, cv2.THRESH_BINARY)

    
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)

    coords = cv2.findNonZero(roi)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = roi[y:y + h, x:x + w]

    if digit.size == 0:
        return None

    digit = cv2.copyMakeBorder(
        digit,
        4, 4, 4, 4,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h, w = digit.shape

    longest = max(w, h)
    scale = 20.0 / longest

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.float32)

    x_start = (28 - new_w) // 2
    y_start = (28 - new_h) // 2

    canvas[
        y_start:y_start + new_h,
        x_start:x_start + new_w
    ] = resized.astype("float32") / 255.0

    return canvas


# Predict one digit

def predict_digit(num):
    """
    num shape: (28, 28)
    """

    if len(model.input_shape) == 4:
        input_data = np.expand_dims(num, axis=(0, -1))  
    elif len(model.input_shape) == 3:
        input_data = np.expand_dims(num, axis=0)      
    else:
        input_data = np.array([num])

    pred = model.predict(input_data, verbose=0)[0]

    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return digit, confidence



# Main loop
while True:
    ret, image = cap.read()

    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    image = cv2.resize(image, (frame_width, frame_height))

 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel3)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel3)

    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)

    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        9
    )


    contour_img = thresh.copy()

    kernel2 = np.ones((2, 2), np.uint8)
    contour_img = cv2.morphologyEx(
        contour_img,
        cv2.MORPH_OPEN,
        kernel2,
        iterations=1
    )


    contour_img = cv2.morphologyEx(
        contour_img,
        cv2.MORPH_CLOSE,
        kernel2,
        iterations=1
    )

    contours, hierarchy = cv2.findContours(
        contour_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        aspect = w / float(h)

        if area < 120:
            continue

        if h < 15:
            continue

        if aspect > 2.5:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    detected_digits = []

    for idx, (x, y, w, h) in enumerate(boxes):
        pad = 8

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_width, x + w + pad)
        y2 = min(frame_height, y + h + pad)

        img_roi = contour_img[y1:y2, x1:x2]

        num = preprocess_digit_for_mnist(img_roi)

        if num is None:
            continue

        digit, confidence = predict_digit(num)

        detected_digits.append(str(digit))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{digit} {confidence:.2f}"

        label_y = y1 - 10
        if label_y < 20:
            label_y = y2 + 25

        cv2.putText(
            image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        debug_num = cv2.resize(
            (num * 255).astype("uint8"),
            (280, 280),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Last Model Input 28x28", debug_num)


    if len(detected_digits) == 0:
        result_text = "Detected: none"
    else:
        result_text = "Detected: " + " ".join(detected_digits)

    cv2.putText(
        image,
        result_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    cv2.imshow("Multi Digit Recognition", image)
    cv2.imshow("Threshold", contour_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
