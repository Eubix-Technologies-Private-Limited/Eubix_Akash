import cv2
import face_recognition

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it
known_image = face_recognition.load_image_file("C:\Eubix\Eubix_Akash\ADAS\face_recognition\dataset\your_name\your_name_0.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
