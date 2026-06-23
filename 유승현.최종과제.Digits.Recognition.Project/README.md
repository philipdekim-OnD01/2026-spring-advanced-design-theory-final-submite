# Raspberry Pi Real-Time Digit Recognition

## 1. Project Overview

This project implements a real-time handwritten digit recognition system using a Raspberry Pi and a USB webcam.
A CNN model trained on the MNIST dataset is used to classify handwritten digits from camera input. The system captures frames from a USB camera, preprocesses the image using OpenCV, detects digit regions through contour detection, and predicts each digit using the trained Keras model.

The final system was deployed and tested on a Raspberry Pi. The model and inference code were uploaded to the Raspberry Pi, and real-time digit recognition was verified using a USB webcam.

## 2. Main Features

* Real-time handwritten digit recognition using a USB webcam
* Raspberry Pi-based on-device AI inference
* MNIST CNN model trained with data augmentation
* OpenCV-based image preprocessing
* Thresholding to separate digits from the background
* Contour detection to detect multiple digits in one frame
* Individual prediction for each detected digit
* Bounding box and prediction result displayed on the screen

## 3. Project Structure

```bash
mnist_camera/
├── mnist_picamera.py
├── train_mnist.py
├── digits_model_best.keras
└── venv/
```

## 4. File Description

### `train_mnist.py`

This file is used to train the digit recognition model.

It loads the MNIST dataset, normalizes the image data, builds a CNN model, and trains the model. During training, the best-performing model based on validation accuracy is automatically saved as:

```bash
digits_model_best.keras
```

The model is saved using `ModelCheckpoint`.

```python
tf.keras.callbacks.ModelCheckpoint(
    "digits_model_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
```

Therefore, `digits_model_best.keras` is not a Python code file. It is the trained model file generated after running `train_mnist.py`.

### `digits_model_best.keras`

This is the trained CNN model file.

It contains the model architecture and trained weights. This file is loaded by the real-time camera inference code.

```python
MODEL_PATH = "digits_model_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)
```

### `mnist_picamera.py`

This is the main real-time inference file.

Although the file name includes `picamera`, the current implementation uses a general USB webcam through OpenCV.

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
```

The script performs the following steps:

1. Open the USB camera.
2. Capture real-time frames.
3. Convert the image to HSV color space.
4. Extract the brightness channel.
5. Apply image preprocessing and thresholding.
6. Detect digit regions using contours.
7. Crop each detected digit.
8. Resize each digit to a 28×28 input image.
9. Predict each digit using `digits_model_best.keras`.
10. Display the predicted result on the camera screen.

## 5. Model Training

To train the model, run:

```bash
cd ~/mnist_camera
source venv/bin/activate
python train_mnist.py
```

After training, the following model file will be created:

```bash
digits_model_best.keras
```

This file is required for real-time digit recognition.

## 6. Real-Time Digit Recognition

To run the camera-based digit recognition system:

```bash
cd ~/mnist_camera
source venv/bin/activate
python mnist_picamera.py
```

Press `q` to quit the camera window.

## 7. Image Preprocessing

The camera image contains background, lighting changes, shadows, and noise. Therefore, preprocessing is required before digit recognition.

The system uses OpenCV thresholding to convert the camera image into a binary image.

```python
_, roi = cv2.threshold(img_roi, 127, 255, cv2.THRESH_BINARY)
```

This converts the image into only two values:

* `0`: black background
* `255`: white digit stroke

As a result, the threshold window shows a black background with only the digit strokes in white. This makes it easier to detect digit contours and prepare the image for MNIST-style input.

## 8. Multiple Digit Recognition

The system can recognize multiple digits in one camera frame.

The process is:

```text
Camera frame
→ Threshold image
→ Find multiple contours
→ Crop each digit region
→ Convert each digit to 28×28
→ Predict each digit separately
→ Display each result on the screen
```

Each detected digit is surrounded by a bounding box, and the predicted digit with confidence score is displayed near the box.

For better recognition, digits should be written with enough spacing between them. If digits are too close, OpenCV may detect them as one connected contour.

## 9. Required Files for Execution

For real-time digit recognition, the two essential files are:

```bash
mnist_picamera.py
digits_model_best.keras
```

If the model needs to be trained again, the training file is also required:

```bash
train_mnist.py
```

Therefore:

```text
Execution only:
- mnist_picamera.py
- digits_model_best.keras

Training + execution:
- train_mnist.py
- digits_model_best.keras
- mnist_picamera.py
```

## 10. Environment

The project was tested on:

```text
Device: Raspberry Pi
Camera: USB webcam
Model framework: TensorFlow / Keras
Image processing: OpenCV
Input image size: 28×28 grayscale
Model file: digits_model_best.keras
```

## 11. Notes

* The USB webcam must be correctly recognized by OpenCV.
* The model file `digits_model_best.keras` must be in the same directory as `mnist_picamera.py`.
* The camera window requires a GUI display environment.
* If running through SSH only, OpenCV GUI windows may not appear.
* The system should be executed directly on the Raspberry Pi display or through a VNC session.

## 12. Result

The trained CNN model and real-time inference code were deployed to the Raspberry Pi. Using a USB webcam, the system successfully detected handwritten digits from the camera image and displayed the predicted result for each digit in real time.

Through this project, the full process of training a digit recognition model, deploying it to Raspberry Pi, processing live camera input, and performing real-time on-device AI inference was implemented and verified.

