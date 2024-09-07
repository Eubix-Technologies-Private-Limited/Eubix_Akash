import cv2
import numpy as np
import time

# Load pre-trained MobileNet SSD model and prototxt file
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Initialize camera
camera = cv2.VideoCapture('crash.mp4')#'dashcam.mp4')
time.sleep(2.0)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Define lane detection parameters
kernel_size = 3
low_threshold = 50
high_threshold = 150
trap_bottom_width = 0.85
trap_top_width = 0.07
trap_height = 0.4
rho = 2
theta = 1 * np.pi/180
threshold = 15
min_line_length = 10
max_line_gap = 20

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None or len(lines) == 0:
        return
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0.:
            slope = 999.
        else:
            slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > 0.5:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)
    right_lines_x = []
    right_lines_y = []
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        right_lines_y.append(y1)
        right_lines_y.append(y2)
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)
    else:
        right_m, right_b = 1, 1
    left_lines_x = []
    left_lines_y = []
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        left_lines_y.append(y1)
        left_lines_y.append(y2)
    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)
    else:
        left_m, left_b = 1, 1
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    if len(right_lines) > 0:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if len(left_lines) > 0:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2

def annotate_image_array(image_in):
    image = filter_colors(image_in)
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    imshape = image.shape
    vertices = np.array([[
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    annotated_image = weighted_img(line_image, image_in)
    return annotated_image

def estimate_distance(box):
    (startX, startY, endX, endY) = box
    box_height = endY - startY
    distance = 1000 / box_height
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

        # Annotate lanes
        annotated_frame = annotate_image_array(frame)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                distance = estimate_distance((startX, startY, endX, endY))
                label = f"Dist: {distance:.2f} feet"

                idx = int(detections[0, 0, i, 1])
                object_label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                obj = object_label.split(":")[0]

                # Check if the object is in the same lane
                if distance < 3 and (startX > frame.shape[1] * (1 - trap_bottom_width) // 2 and endX < frame.shape[1] - (frame.shape[1] * (1 - trap_bottom_width) // 2)):
                    print(f"{obj} near vehicle at distance {distance:.2f} feet")

                    cv2.putText(annotated_frame, "WARNING: COLLISION RISK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Draw the bounding box and label on the frame
                cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Vehicle Collision Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    camera.release()
    cv2.destroyAllWindows()
