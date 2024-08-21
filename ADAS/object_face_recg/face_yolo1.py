import cv2
import numpy as np
import face_recognition
import os

# Load YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize lists for known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Define the dataset directory path
dataset_dir = "dataset/"

# Load face encodings and names from the dataset folder
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

print(f"Loaded {len(known_face_encodings)} face encodings from the dataset.")

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
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            
            if class_names[class_ids[i]] == "person":
                face_image = frame[y:y+h, x:x+w]
                face_encodings = face_recognition.face_encodings(face_image)

                if face_encodings:  # Ensure that face_encodings is not empty
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    else:
                        name = "unknown person"
                    
                    label = f"{name}: {confidences[i]:.2f}"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame



# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    detected_frame = detect_objects(frame)
    
    # Display the result
    cv2.imshow('YOLOv4 Object Detection', detected_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
