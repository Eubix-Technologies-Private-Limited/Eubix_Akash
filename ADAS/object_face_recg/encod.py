import face_recognition
import os
import pickle

# Initialize lists for known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Define the dataset directory path
dataset_dir = "dataset/"

# Loop through each person in the dataset directory
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    if os.path.isdir(person_dir):
        # Loop through each image file in the person's directory
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

# Save the encodings and names to a file
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print(f"Saved {len(known_face_encodings)} face encodings.")
