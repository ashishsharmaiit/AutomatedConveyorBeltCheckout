from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import time

# Replace 'YOUR_ENDPOINT' and 'YOUR_KEY' with your Azure endpoint and Computer Vision subscription key.
endpoint = 'https://pickyrobotics.cognitiveservices.azure.com/'
key = '58ce4558b00b4c25a0340ca78808bb59'

# Authenticate the client with your subscription key and endpoint
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

# Specify the image URL or path
image_url = '1.5.jpg'

# Call the API to extract the text
read_response = computervision_client.read_in_stream(open(image_url, "rb"),  raw=True)

# Get the operation location (URL with an ID at the end) from the response
operation_location = read_response.headers["Operation-Location"]

# Grab the ID from the URL
operation_id = operation_location.split("/")[-1]

# Wait for the read operation to complete
while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status.lower() not in ['notstarted', 'running']:
        break
    time.sleep(1)  # wait for a second before querying the service again

# Print the detected text, line by line
if read_result.status.lower() == 'succeeded':
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            print(line.text)
            print(line.bounding_box)
