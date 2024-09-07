import cv2
import numpy as np
import time

# Load pre-trained MobileNet SSD model and prototxt file
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Initialize video capture with the video file
camera = cv2.VideoCapture('dashcam.mp4')
time.sleep(2.0)
if not camera.isOpened():
    print("Error: Could not open video.")


# List of class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def estimate_distance(box):
    # Estimate distance to the detected object using the bounding box size
    (startX, startY, endX, endY) = box
    box_height = endY - startY
    distance = 1000 / box_height  # Placeholder formula, adjust based on calibration
    return distance

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break  # Exit the loop if the video ends

        # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Get the bounding box for the detected object
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                print("Frame read successfully") if ret else print("Failed to read frame")


                # Estimate distance to the object
                distance = estimate_distance((startX, startY, endX, endY))
                label = f"Dist: {distance:.2f} feet"

                idx = int(detections[0, 0, i, 1])
                object = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                obj = object.split(":")[0]
                
                # Print warning if an object is near the vehicle
                if distance < 4:
                    print(f"{obj} near vehicle at distance {distance:.2f} feet")

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for collision risk
                if distance < 4:  # Example threshold for triggering alert
                    cv2.putText(frame, "WARNING: COLLISION RISK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display the frame
        cv2.imshow("Vehicle Collision Detection", frame)


        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()
