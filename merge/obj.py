import cv2
import numpy as np
import os
import time

# Paths to model files
weights_path = r"yolov7-tiny.weights"
config_path = r"yolov7-tiny.cfg"
names_path = r"coco.names"

# Ensure paths exist
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"Class names file not found: {names_path}")

# Load YOLOv4
net = cv2.dnn.readNet(weights_path, config_path)

# Load the COCO class labels
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Set preferable backend to CUDA if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers()
if isinstance(out_layer_indices[0], list):
    out_layer_indices = [i[0] for i in out_layer_indices]
output_layers = [layer_names[i - 1] for i in out_layer_indices]

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            if isinstance(detection, np.ndarray) and detection.shape[0] > 5:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detected_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"  # Use class name
            detected_objects.append((x, y, w, h, label))
    return detected_objects

# Open webcam
cap = cv2.VideoCapture(0)

# Frame skipping rate
#skip_rate = 1
#frame_count = 0
prev_time = time.time()

# Initialize previous detections
previous_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the current time
    current_time = time.time()

    # Skip frames based on the skip_rate
    #if frame_count % skip_rate == 0:
        # Perform object detection
    detected_objects = detect_objects(frame)
        
        # Update previous detections
    #previous_detections = detected_objects

    #frame_count += 1

    # Draw previous detections on the frame
    for (x, y, w, h, label) in detected_objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate and display FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with FPS
    cv2.imshow('YOLOv4 Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()