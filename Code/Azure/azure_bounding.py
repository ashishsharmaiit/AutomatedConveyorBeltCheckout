from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import time
import cv2
import numpy as np
import random

# Function to generate a random color
def generate_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Azure credentials and client setup
endpoint = 'https://pickyrobotics.cognitiveservices.azure.com/'
key = '58ce4558b00b4c25a0340ca78808bb59'
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

# Specify the image path
image_path = 'frame_71.jpg'

# Call the API to extract the text
with open(image_path, "rb") as image_stream:
    read_response = computervision_client.read_in_stream(image_stream, raw=True)

# Get the operation location (URL with an ID at the end) from the response
operation_location = read_response.headers["Operation-Location"]
operation_id = operation_location.split("/")[-1]

# Wait for the read operation to complete
while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status.lower() not in ['notstarted', 'running']:
        break
    time.sleep(1)

# Load the image using OpenCV
image = cv2.imread(image_path)

## ... [previous code] ...

# Check the result from Azure OCR and process if successful
if read_result.status.lower() == 'succeeded':
    elements = []  # To store the bounding box and text
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            # Add each line's text and bounding box to elements list
            elements.append((line.bounding_box, line.text))
            print(line.text)
            print(line.bounding_box)

    # Draw bounding boxes and annotations
    for element in elements:
        bbox, text = element
        color = generate_color()  # Generate a random color for each bounding box
        # Draw the bounding box
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        # Put the text annotation near the bounding box
        cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Define a new filename for the annotated image
    # Splitting the original filename and adding "_annotated" suffix before the file extension
    base_filename, file_extension = image_path.rsplit('.', 1)
    new_filename = f"{base_filename}_annotated.{file_extension}"

    # Save the annotated image
    cv2.imwrite(new_filename, image)

    # Show the image with all bounding boxes and annotations
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Text reading didn't succeed.")
