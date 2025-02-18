# Object Detection using TensorFlow and OpenCV

## 📌 Overview

This project implements object detection using a pre-trained **SSD MobileNet v2** model from the TensorFlow Model Zoo. The model is loaded using OpenCV's Deep Neural Network (DNN) module, and it performs real-time object detection on images.

## ⚠️ Disclaimer

**This project is intended for educational purposes only. The author is not responsible for any misuse.**

## 🚀 Features

- Loads and processes images for object detection.
- Uses a pre-trained **SSD MobileNet v2** model.
- Detects multiple objects in an image and labels them with bounding boxes.
- Configurable detection threshold.

## 📂 Requirements

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib
- numpy
- os
- urllib
- zipfile
- TensorFlow Model Zoo (pre-trained models)

## 🔧 Installation

1. Clone this repository:
   ```sh
   git remote add origin https://github.com/konynour/TF-Object-Detection.git   ```
2. Navigate to the project folder:
   ```sh
   cd ObjectDetection
   ```
3. Install dependencies:
   ```sh
   pip install opencv-python numpy matplotlib
   ```

## ⚙️ Usage

### 1️⃣ Load and Configure the Model

- Download and extract the **SSD MobileNet v2** model from TensorFlow Model Zoo.
- Place the `frozen_inference_graph.pb` and its configuration file `.pbtxt` inside the `models/` directory.

### 2️⃣ Run Object Detection on an Image

```python
import cv2
import os
from detect import detect_objects, display_objects

# Load an image
im = cv2.imread(os.path.join("images", "street.jpg"))

# Detect objects
objects = detect_objects(net, im)

# Display results
display_objects(im, objects)
```

### 3️⃣ Adjust Detection Threshold

If you want to adjust the confidence threshold, you can modify the detection function call:

```python
im = cv2.imread(os.path.join("images", "baseball.jpg"))
objects = detect_objects(net, im)
display_objects(im, objects, threshold=0.2)
```



