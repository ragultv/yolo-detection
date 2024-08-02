from ultralytics import YOLO
from PIL import Image
import numpy as np
from google.colab.patches import cv2_imshow
import cv2
from paddleocr import PaddleOCR

# Load the YOLO model
model = YOLO("best.pt")

def predict(img):
    return model.predict(img)

def extract_license_plate_text(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(img)
    print(result)
    if result and len(result) > 0:
        return result[0][0][1][0]  # Extracting the text part from the result
    return ""

# Read the image
img = cv2.imread('car7.jpg')

# Run YOLO model on the image
results = predict(img)

for detection in results[0].boxes:
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    license_plate_img = img[y1:y2, x1:x2]

    # Extract text from the license plate image
    license_number = extract_license_plate_text(license_plate_img)

    # Draw rectangle and put text on the image
    cv2.putText(img, license_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Display the image with annotations
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
