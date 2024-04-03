import cv2
import numpy as np
import random

# Load the image
image = cv2.imread('1.5.jpg')

# List of bounding box coordinates and their respective texts
# Each item in the list is a tuple (bounding_box, text)
elements = [
    ([44.0, 423.0, 46.0, 393.0, 55.0, 393.0, 54.0, 423.0], "undry"),
    ([126.0, 430.0, 126.0, 360.0, 137.0, 361.0, 136.0, 431.0], "essentials"),
    ([148.0, 293.0, 125.0, 252.0, 140.0, 243.0, 165.0, 281.0], "Kroger"),
    ([211.0, 306.0, 136.0, 188.0, 162.0, 173.0, 237.0, 290.0], "DECAF"),
    ([248.0, 292.0, 166.0, 163.0, 194.0, 145.0, 276.0, 275.0], "GREEN"),
    ([277.0, 269.0, 199.0, 149.0, 247.0, 117.0, 326.0, 243.0], "TEA"),
    ([351.0, 190.0, 303.0, 112.0, 323.0, 100.0, 371.0, 178.0], "NET WT 1.9 OZ (54g)"),
    ([407.0, 267.0, 396.0, 248.0, 402.0, 245.0, 413.0, 263.0], "BAGS")
]

# Function to generate a random color
def generate_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Loop through all the elements and draw each bounding box and text annotation on the image
for element in elements:
    bbox, text = element
    color = generate_color()  # Generate a random color for each bounding box

    # Draw the bounding box
    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    # Put the text annotation near the bounding box
    cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with all bounding boxes and annotations
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
