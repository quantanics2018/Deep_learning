# Import necessary libraries
import cv2
import numpy as np
from math import pow, sqrt

# Configuration parameters
preprocessing = False
calculateConstant_x = 300
calculateConstant_y = 615
personLabelID = 15.00
debug = True
accuracyThreshold = 0.4
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
write_video = False

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def CLAHE(bgr_image: np.array) -> np.array:
    # Convert BGR image to HSV color space
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    # Apply CLAHE to the value channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    # Convert back to BGR color space
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to calculate centroid of a bounding box
def centroid(startX, endX, startY, endY):
    centroid_x = round((startX + endX) / 2, 4)
    centroid_y = round((startY + endY) / 2, 4)
    bboxHeight = round(endY - startY, 4)
    return centroid_x, centroid_y, bboxHeight

# Function to calculate distance based on bounding box height
def calcDistance(bboxHeight):
    distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
    return distance

# Function to draw bounding boxes on the frame based on risk level
def drawResult(frame, position):
    for i in position.keys():
        if i in highRisk:
            rectangleColor = RED
        elif i in mediumRisk:
            rectangleColor = YELLOW
        else:
            rectangleColor = GREEN
        (startX, startY, endX, endY) = detectionCoordinates[i]
        cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)

# Main function
if __name__ == "__main__":
    # Load pre-trained SSD MobileNet model
    caffeNetwork = cv2.dnn.readNetFromCaffe(".\Model\SSD_MobileNet_prototxt.txt", ".\Model\SSD_MobileNet.caffemodel")
    # Open video file
    cap = cv2.VideoCapture("Video.mp4")

    while cap.isOpened():
        # Read frame from the video
        debug_frame, frame = cap.read()
        highRisk = set()
        mediumRisk = set()
        position = dict()
        detectionCoordinates = dict()

        # Check for the end of the video or if unable to open the video
        if not debug_frame:
            print("End of video or unable to open the video!")
            break

        # Apply preprocessing if enabled
        if preprocessing:
            frame = CLAHE(frame)

        # Get the dimensions of the frame
        (imageHeight, imageWidth) = frame.shape[:2]
        # Prepare the frame for object detection
        pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

        # Set the input to the pre-trained model
        caffeNetwork.setInput(pDetection)
        # Forward pass to get detections
        detections = caffeNetwork.forward()

        # Process each detection
        for i in range(detections.shape[2]):
            accuracy = detections[0, 0, i, 2]
            if accuracy > accuracyThreshold:
                # Get the class and box coordinates
                idOfClasses = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                (startX, startY, endX, endY) = box.astype('int')

                # Check if the detected object is a person
                if idOfClasses == personLabelID:
                    # Draw default bounding box
                    bboxDefaultColor = (255, 255, 255)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bboxDefaultColor, 2)
                    detectionCoordinates[i] = (startX, startY, endX, endY)

                    # Calculate centroid and distance in centimeters
                    centroid_x, centroid_y, bboxHeight = centroid(startX, endX, startY, endY)
                    distance = calcDistance(bboxHeight)
                    centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                    centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                    position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)

        # Evaluate risk levels based on distance between centroids
        for i in position.keys():
            for j in position.keys():
                if i < j:
                    distanceOfBboxes = sqrt(pow(position[i][0] - position[j][0], 2)
                                            + pow(position[i][1] - position[j][1], 2)
                                            + pow(position[i][2] - position[j][2], 2))
                    if distanceOfBboxes < 150:  # 150cm or lower
                        highRisk.add(i), highRisk.add(j)
                    elif 150 < distanceOfBboxes < 200:  # between 150 and 200
                        mediumRisk.add(i), mediumRisk.add(j)

        # Display risk information on the frame
        cv2.putText(frame, "Person in High Risk : " + str(len(highRisk)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Person in Medium Risk : " + str(len(mediumRisk)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
        cv2.putText(frame, "Detected Person : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # Draw bounding boxes on the frame based on risk level
        drawResult(frame, position)

        # Display the resulting frame
        cv2.imshow('Result', frame)
        # Check for the 'q' key to quit the application
        waitkey = cv2.waitKey(1)
        if waitkey == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
