import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

# Load the class labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(rgb_frame, size=(300, 300), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    # Process each detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            class_label = labels[class_id]
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = box.astype(int)

            # Draw a red box around the detected object
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

            # Display the class label and confidence
            text = f'{class_label}: {confidence:.2f}'
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
