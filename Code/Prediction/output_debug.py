import os
import cv2
import numpy as np
import random
import time
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
from pymongo import MongoClient
from operator import itemgetter
from bson import ObjectId

# Helper function to convert ObjectId to str
def json_convert(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


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
    threshold = 0.5
    detections_list = []  # Use a list to collect detections
    results = model(frame)[0]
    box_id = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            class_name = results.names[int(class_id)]
            box_id += 1
            new_detection = {'BoxID': box_id, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}
            detections_list.append(new_detection)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"{class_name.upper()} {box_id}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    detection_results = {'detections': detections_list}
    cv2.imwrite('predicted_boxes.jpg', frame)
    return detection_results


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
        for element in elements:
            bbox, text = element
            color = generate_color()
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite('Annotated_Another_Image.jpg', image)
    
    # Instead of writing to an Excel file, save text elements in a dictionary
    text_elements = {'text_extraction': [{'bbox': bbox, 'text': text} for bbox, text in elements]}
    cv2.imwrite('Text_annotated_predicted.jpg', image)
    return text_elements

# Combining data
def combine_data(detections_dict, texts_dict):
    products_data = detections_dict['detections']
    texts_data = texts_dict['text_extraction']
    mapped_data = []
    for text_element in texts_data:
        text_bbox = text_element['bbox']
        text_area = calculate_text_area(text_bbox)
        text_length = len(text_element['text'])
        for product_data in products_data:
            product_bbox = {'X1': product_data['X1'], 'Y1': product_data['Y1'], 'X2': product_data['X2'], 'Y2': product_data['Y2']}
            product_area = calculate_box_area([product_bbox['X1'], product_bbox['Y1'], product_bbox['X2'], product_bbox['Y2']])
            if is_inside(text_bbox, product_bbox):
                area_ratio = text_area / product_area if product_area != 0 else 0
                area_ratio_per_character = area_ratio / text_length if text_length != 0 else 0
                combined_data = {
                    'BoxID': product_data['BoxID'],
                    'Text': text_element['text'],
                    'Area_Ratio_Per_Character': area_ratio_per_character
                }
                mapped_data.append(combined_data)
                break

    return {'combined_data': mapped_data}

def get_top_words(combined_data, top_n=3):
    """
    Get top N words for each BoxID based on Area_Ratio_Per_Character.
    """
    top_words = {}
    for item in combined_data:
        box_id = item['BoxID']
        if box_id not in top_words:
            top_words[box_id] = []
        top_words[box_id].append(item)
    for box_id, words in top_words.items():
        # Sort the words based on Area_Ratio_Per_Character and get the top N
        top_words[box_id] = sorted(words, key=itemgetter('Area_Ratio_Per_Character'), reverse=True)[:top_n]
    return top_words

def find_product_name(top_words, products_collection):
    """
    Query MongoDB for the top words and calculate scores, then find the product with the highest score.
    Include detailed information for each step of the process.
    """
    product_mapping = {}
    debug_info = {}  # This will store the detailed information for debugging

    for current_box_id, words in top_words.items():
        product_scores = {}
        debug_info[current_box_id] = {'Top_Words': [], 'Search_Results': {}, 'Scores': {}}

        for word_info in words:
            text = word_info['Text']
            area_ratio_per_character = word_info['Area_Ratio_Per_Character']
            debug_info[current_box_id]['Top_Words'].append({'Text': text, 'Area_Ratio_Per_Character': area_ratio_per_character})
            
            # Search MongoDB for the text
            cursor = products_collection.find({'Sides.Texts.Text': text})
            search_results = list(cursor)
            debug_info[current_box_id]['Search_Results'][text] = search_results
            
            # Calculate score for each matching word
            for doc in search_results:
                mongodb_box_id = str(doc['BoxID'])  # Convert ObjectId to string
                if mongodb_box_id not in product_scores:
                    product_scores[mongodb_box_id] = 0
                    debug_info[current_box_id]['Scores'][mongodb_box_id] = []

                for side in doc['Sides']:
                    for text_entry in side['Texts']:
                        if text_entry['Text'] == text:
                            score = text_entry['Area_per_character']
                            product_scores[mongodb_box_id] += score
                            # Store the detailed info
                            debug_info[current_box_id]['Scores'][mongodb_box_id].append({
                                'Text': text,
                                'MongoDB_Area_per_character': score,
                                'Current_Area_Ratio_Per_Character': area_ratio_per_character,
                                'Score_Added': score
                            })
                            break

        # If we found matching products, find the one with the highest score
        if product_scores:
            best_match = max(product_scores, key=product_scores.get)
            product_mapping[current_box_id] = best_match
        else:
            product_mapping[current_box_id] = None  # or some placeholder to indicate no match was found

    return product_mapping, debug_info


# Main execution
if __name__ == '__main__':
    image_path = 'IMG_7141.jpg' 
    model_path = os.path.join('..', 'BoxModel', 'runs', 'detect', 'train2', 'weights', 'last.pt')
    endpoint = 'https://pickyrobotics.cognitiveservices.azure.com/'
    key = '58ce4558b00b4c25a0340ca78808bb59'
    client = MongoClient('localhost', 27017)
    db = client['productDB']
    products_collection = db['products']

    detections = process_object_detection(image_path, model_path)
    text_elements = process_text_extraction(image_path, endpoint, key)
    combined_results = combine_data(detections, text_elements)

    top_words = get_top_words(combined_results['combined_data'])
    product_mapping, debug_info = find_product_name(top_words, products_collection)

    # Output the mapping of current product BoxID to MongoDB BoxID and debug info
    output = {
        'Product_Mapping': [{'Current_BoxID': k, 'MongoDB_BoxID': v} for k, v in product_mapping.items()],
        'Debug_Info': debug_info
    }

    serialized_output = json.dumps(debug_info, indent=4, default=json_convert)

    # Save the serialized output to a log file
    log_file_path = 'debug_log.txt'
    with open(log_file_path, 'w') as file:
        file.write(serialized_output)
