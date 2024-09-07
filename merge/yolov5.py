import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small version, use 'yolov5m' or 'yolov5l' for larger models

# Open camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    results = model(frame)
    
    # Render results on the frame
    annotated_frame = results.render()[0]
    
    # Display the frame
    cv2.imshow('Object Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
