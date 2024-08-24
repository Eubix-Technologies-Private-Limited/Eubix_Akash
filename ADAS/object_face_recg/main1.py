import dlib
import cv2
import numpy as np
import pickle

# Load known face encodings and names
with open('known_face_encodings.pkl', 'rb') as f:
    known_face_encodings = pickle.load(f)

with open('encodings1.pkl', 'rb') as f:
    known_face_names = pickle.load(f)

# Initialize dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load image to check
image = cv2.imread('dataset\Akash\Akash_0.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

for face in faces:
    # Get face landmarks
    shape = sp(gray, face)
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    face_encoding = np.array(face_descriptor)

    # Compare the face encoding with known encodings
    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
    min_distance_index = np.argmin(distances)
    
    if distances[min_distance_index] < 0.6:  # Threshold for matching
        name = known_face_names[min_distance_index]
    else:
        name = "Unknown"

    # Draw a rectangle around the face and label it
    (x, y, w, h) = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    cv2.putText(image, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# Display the result
cv2.imshow("Image with Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
