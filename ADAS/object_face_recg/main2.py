import cv2
import numpy as np
import dlib
import os
import pickle

# Load YOLOv4
net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers()
if isinstance(out_layer_indices[0], list):
    out_layer_indices = [i[0] for i in out_layer_indices]
output_layers = [layer_names[i - 1] for i in out_layer_indices]

# Initialize dlib's face detector and the face recognizer model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory containing the dataset
dataset_dir = "dataset"
encodings_file = "known_face_encodings.pkl"

def get_face_encoding(image):
    dets = detector(image, 1)
    if len(dets) > 0:
        shape = predictor(image, dets[0])
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    return None

def load_encodings():
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    return [], []

def save_encodings(encodings, names):
    with open(encodings_file, "wb") as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)

# Load known face encodings
known_face_encodings, known_face_names = load_encodings()

# Load the dataset if encodings are not available
if not known_face_encodings:
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize image to speed up processing
                    small_image = cv2.resize(image, (128, 128))
                    encoding = get_face_encoding(small_image)
                    if encoding is not None:
                        known_face_encodings.append(encoding)
                        known_face_names.append(person_name)
    # Save encodings after loading dataset
    save_encodings(known_face_encodings, known_face_names)

def detect_objects_and_recognize_faces(frame):
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
                person_image = frame[y:y+h, x:x+w]
                
                if person_image.size == 0:
                    print(f"Warning: Detected person image is empty at coordinates ({x}, {y}, {w}, {h})")
                    continue

                # Resize the person image to speed up face recognition
                small_person_image = cv2.resize(person_image, (128, 128))

                # Detect faces in the person image
                faces = detector(small_person_image, 1)

                for face in faces:
                    shape = predictor(small_person_image, face)
                    face_encoding = np.array(face_recognizer.compute_face_descriptor(small_person_image, shape))

                    # Compare with known faces
                    matches = []
                    for known_encoding in known_face_encodings:
                        distance = np.linalg.norm(known_encoding - face_encoding)
                        matches.append(distance)

                    # Find the best match
                    if matches:
                        min_distance = min(matches)
                        if min_distance < 0.6:  # Threshold for a good match
                            best_match_index = matches.index(min_distance)
                            name = known_face_names[best_match_index]
                        else:
                            name = "Unknown"

                        # Draw the name of the person
                        cv2.rectangle(person_image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(person_image, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
skip_frames = 2  # Perform face detection every 2 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1

    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (640, 480))  # Maintain reasonable resolution for YOLOv4

    if frame_count % skip_frames == 0:
        # Perform object detection and face recognition
        detected_frame = detect_objects_and_recognize_faces(small_frame)
    else:
        detected_frame = small_frame

    # Display the result
    cv2.imshow('YOLOv4 + Dlib Face Recognition', detected_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
