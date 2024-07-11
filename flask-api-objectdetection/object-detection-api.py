import torch
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model_path = 'yolov8n.pt'  # Path to the model file
model = YOLO(model_path)  # Load the model

# Load COCO dataset classes
coco_classes = model.names

# Define a function to perform object detection
def detect_objects(image):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.float() / 255.0

    # Perform object detection
    results = model(image)

    # Extract bounding boxes and class labels
    boxes = []
    labels = []
    scores = []
    for det in results.xyxy[0].tolist():
        scores.append(det[4])
        labels.append(coco_classes[int(det[5])])
        boxes.append([int(det[0]), int(det[1]), int(det[2]), int(det[3])])

    return boxes, labels, scores

# Define a Flask endpoint to receive images and perform object detection
@app.route('/detect', methods=['POST'])
def detect():
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    boxes, labels, scores = detect_objects(image)

    # Return the detection results
    return jsonify({'boxes': boxes, 'labels': labels, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)