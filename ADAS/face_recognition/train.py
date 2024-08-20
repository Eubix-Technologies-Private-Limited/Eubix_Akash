import os
import face_recognition
import cv2
import numpy as np

# Path to the dataset
dataset_path = "./dataset/"

# List to hold encodings and labels
known_face_encodings = []
known_face_names = []

# Load each person's images and extract face encodings
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_dir):
        continue
    
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        
        # Load the image
        img = face_recognition.load_image_file(img_path)
        
        # Get the face encoding
        face_encoding = face_recognition.face_encodings(img)
        
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(person_name)

# Save the encodings and names for later use
np.savez("face_encodings.npz", encodings=known_face_encodings, names=known_face_names)

print("Model trained and saved.")