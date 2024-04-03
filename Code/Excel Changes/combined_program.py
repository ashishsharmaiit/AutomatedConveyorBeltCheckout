import os
import cv2
import pandas as pd
import numpy as np
import random
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
from openpyxl import Workbook
from openpyxl.styles import NamedStyle
from openpyxl.utils import get_column_letter

# Function to generate a random color
def generate_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Function to calculate the area of the bounding box
def calculate_text_area(bbox):
    # Assuming bbox is [x1, y1, x2, y2, x3, y3, x4, y4]
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    # Calculate area using the Shoelace formula
    area = 0.5 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))
    return area


# Function to check if the text is inside the product bounding box
def is_inside(text_box, product_box):
    return (min(text_box[0::2]) >= product_box['X1'] and max(text_box[0::2]) <= product_box['X2'] and
            min(text_box[1::2]) >= product_box['Y1'] and max(text_box[1::2]) <= product_box['Y2'])

def calculate_box_area(bbox):
    width = max(bbox[0::2]) - min(bbox[0::2])
    height = max(bbox[1::2]) - min(bbox[1::2])
    return width * height

# Object detection with YOLO
def process_object_detection(image_path, model_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to read image from {image_path}")

    model = YOLO(model_path)
    threshold = 0.3
    results_df = pd.DataFrame(columns=['BoxID', 'X1', 'Y1', 'X2', 'Y2'])
    results = model(frame)[0]
    box_id = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            class_name = results.names[int(class_id)]
            box_id += 1
            new_row = {'BoxID': box_id, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"{class_name.upper()} {box_id}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    if not cv2.imwrite('IMG_7134_with_boxes.jpg', frame):
        raise ValueError("Failed to write image 'IMG_7134_with_multiple_boxes.jpg'")
    results_df.to_excel('detected_boxes.xlsx', index=False)

# Text extraction with Azure
def process_text_extraction(image_path, endpoint, key):
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
    with open(image_path, "rb") as image_stream:
        read_response = computervision_client.read_in_stream(image_stream, raw=True)
    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(1)
    image = cv2.imread(image_path)
    if read_result.status.lower() == 'succeeded':
        elements = []
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                elements.append((line.bounding_box, line.text))
        df = pd.DataFrame(elements, columns=['Bounding Box', 'Text'])
        df.to_excel('azure_output.xlsx', engine='xlsxwriter')
        for element in elements:
            bbox, text = element
            color = generate_color()
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite('Annotated_Another_Image.jpg', image)

# Combining Excel files
def combine_excel_files(product_file_path, text_file_path, output_file_path):
    products_df = pd.read_excel(product_file_path)
    texts_df = pd.read_excel(text_file_path)
    texts_df['Bounding Box'] = texts_df['Bounding Box'].apply(eval)
    product_area_ratios = {}
    mapped_data = []
    for _, text_row in texts_df.iterrows():
        text_bbox = text_row['Bounding Box']
        text_area = calculate_text_area(text_bbox)
        text_length = len(text_row['Text'])
        for _, product_row in products_df.iterrows():
            product_bbox = {'X1': product_row['X1'], 'Y1': product_row['Y1'], 'X2': product_row['X2'], 'Y2': product_row['Y2']}
            product_area = calculate_box_area([product_bbox['X1'], product_bbox['Y1'], product_bbox['X2'], product_bbox['Y2']])
            if is_inside(text_bbox, product_bbox):
                area_ratio = text_area / product_area if product_area != 0 else 0
                area_ratio_per_character = area_ratio / text_length if text_length != 0 else 0
                combined_data = {
                    'BoxID': product_row['BoxID'],
                    'Text': text_row['Text'],
                    'Area_Ratio_Per_Character': area_ratio_per_character
                }
                mapped_data.append(combined_data)
                break

    combined_df = pd.DataFrame(mapped_data)
    combined_df = combined_df[['BoxID', 'Text', 'Area_Ratio_Per_Character']]  # Select only required columns
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        combined_df.to_excel(writer, index=False)

# Main execution
if __name__ == '__main__':
    image_path = 'IMG_7140.JPG'
    model_path = os.path.join('.', 'BoxModel', 'runs', 'detect', 'train2', 'weights', 'last.pt')
    endpoint = 'https://pickyrobotics.cognitiveservices.azure.com/'
    key = '58ce4558b00b4c25a0340ca78808bb59'

    process_object_detection(image_path, model_path)
    process_text_extraction(image_path, endpoint, key)
    combine_excel_files('detected_boxes.xlsx', 'azure_output.xlsx', 'Combined_database.xlsx')
