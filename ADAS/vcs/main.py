import cv2
import numpy as np
import time

# Load pre-trained MobileNet SSD model and prototxt file
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Initialize camera
camera = cv2.VideoCapture(1)
time.sleep(2.0)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

def estimate_distance(box):
    # Estimate distance to the detected object using the bounding box size
    # This is a placeholder formula. You may need to calibrate it based on your camera setup.
    (startX, startY, endX, endY) = box
    box_height = endY - startY
    distance = 1000 / box_height  # Example: adjust based on real-world testing
    return distance

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

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

                # Estimate distance to the object
                distance = estimate_distance((startX, startY, endX, endY))
                label = f"Dist: {distance:.2f} feet"

                idx = int(detections[0, 0, i, 1])
                object = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
                obj = object.split(":")[0]
                if distance < 4:
                    print(f"{obj} near vehicle at distance {distance:.2} feet")

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for collision risk (example threshold distance)
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