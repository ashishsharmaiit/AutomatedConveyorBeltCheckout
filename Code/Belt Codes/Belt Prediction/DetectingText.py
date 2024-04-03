import os
import cv2
import numpy as np
import random
import time
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

class AzureTextIdentifier:
    def __init__(self, endpoint, key):
        self.computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    @staticmethod
    def generate_color():
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def process_text_extraction(self, image_path, test_mode=False):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        directory = os.path.dirname(image_path)

        json_filename = os.path.join(directory, f'{base_name}_detected_text.json')
        annotated_image_path = os.path.join(directory, f'{base_name}_annotated.jpg')

        if test_mode:
            return self.mock_text_extraction_response(json_filename)
        
        with open(image_path, "rb") as image_stream:
            read_response = self.computervision_client.read_in_stream(image_stream, raw=True)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        while True:
            read_result = self.computervision_client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            time.sleep(1)
        image = cv2.imread(image_path)
        if read_result.status.lower() == 'succeeded':
            elements = []
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    elements.append((line.bounding_box, line.text))
            for element in elements:
                bbox, text = element
                color = self.generate_color()
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(annotated_image_path, image)
        
        
        # Instead of writing to an Excel file, save text elements in a dictionary
        text_elements = {'text_extraction': [{'bbox': bbox, 'text': text} for bbox, text in elements]}
        with open(json_filename, 'w') as file:
            json.dump(text_elements, file)
        return text_elements

    def mock_text_extraction_response(self, json_filename):
        try:
            with open(json_filename, 'r') as file:
                static_response = json.load(file)
            return static_response
        except FileNotFoundError:
            print(f"Mock response file not found at {json_filename}. Returning empty response.")
            return {'text_extraction': []}
