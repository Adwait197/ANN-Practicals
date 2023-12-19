import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained model
model = tf.saved_model.load('path/to/saved_model')

# Load the label map
label_map = {1: 'person', 2: 'car', 3: 'dog'}  # Replace with your label map

# Load and preprocess the input image
image = cv2.imread('path/to/input_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.expand_dims(image, axis=0)
image = tf.convert_to_tensor(image, dtype=tf.float32)

# Run object detection
detections = model(image)
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(int)

# Filter detections based on score threshold
threshold = 0.5  # Adjust as per your needs
filtered_boxes = boxes[scores >= threshold]
filtered_classes = classes[scores >= threshold]

# Display the detected objects
for box, cls in zip(filtered_boxes, filtered_classes):
    ymin, xmin, ymax, xmax = box
    ymin *= image.shape[1]
    xmin *= image.shape[2]
    ymax *= image.shape[1]
    xmax *= image.shape[2]

    label = label_map[cls]
    cv2.rectangle(image[0], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.putText(image[0], label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Object Detection', image[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
