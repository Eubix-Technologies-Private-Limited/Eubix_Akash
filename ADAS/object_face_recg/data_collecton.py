import cv2
import os

# Define the name of the person
person_name = input()
dataset_path = "./dataset/" + person_name

# Create the directory if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Start the webcam
cap = cv2.VideoCapture(0)

# Initialize a counter for the number of images captured
img_count = 0

print("Press 'q' to stop capturing images.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Display the frame
    cv2.imshow("Capturing Face Data", frame)
    
    # Save the frame every 5 frames
    if img_count % 5 == 0:
        img_name = os.path.join(dataset_path, f"{person_name}_{img_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
    
    img_count += 1
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print("Face data collection complete.")
