import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model_path = "yolov8n.pt"
model = YOLO(model_path)

# Load an image
image_path = 'C:/Users/tragu/Downloads/projects/data/images/train/001a23c9dd973787.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}.")
    exit()

# Predict with the model
results = model(image)

# Print raw results for debugging
print("Raw results:", results)

# Set a lower confidence threshold to ensure detections are not missed
confidence_threshold = 0.2

# Visualize the results on the image
for result in results:
    for box in result.boxes:
        # Extract box coordinates and class id
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = box.conf[0]

        # Print box coordinates, class id, and confidence for debugging
        print(f"Box coordinates: {(x1, y1, x2, y2)}, Class ID: {class_id}, Confidence: {confidence}")

        # Check if the detected class is the object of interest (adjust class_id as needed)
        # Also check if the confidence is above the threshold
        if confidence > confidence_threshold:
            # Draw rectangle around the detected object
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Optionally, add a label
            label = f"dog: {confidence:.2f}"  # Confidence score
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with detections
cv2.imshow('YOLO Detection', image)
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()
