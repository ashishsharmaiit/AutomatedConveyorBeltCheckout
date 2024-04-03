import pandas as pd
import math
from collections import Counter


# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to deactivate old objects if not seen recently or near the exit point
def deactivate_old_objects(current_frame, objects, max_frames_since_last_seen, exit_threshold_y):
    for obj_id, obj_data in list(objects.items()):
        frames_since_seen = current_frame - obj_data['last_frame_seen']
        _, last_center_y = obj_data['last_position']
        # Check if the object hasn't been seen recently or is near the y exit
        if frames_since_seen > max_frames_since_last_seen and last_center_y <= exit_threshold_y:
            obj_data['active'] = False


# Load your data from both models
data1 = pd.read_excel('belt_detected_products.xlsx')
data2 = pd.read_excel('belt_detected_boxes.xlsx')

# Add a source column to each DataFrame before concatenation
data1['Source'] = 'Product'
data2['Source'] = 'Box'

MAX_FRAMES_SINCE_LAST_SEEN = 20

# Combine the data from both models into one DataFrame
combined_data = pd.concat([data1, data2])

# Sort the combined data by the 'Frame' column to process the frames in order
combined_data.sort_values(by='Frame', inplace=True)

# Threshold for considering objects as the same
DISTANCE_THRESHOLD = 100  # Adjust this based on your specific use case

# Dictionary to track class occurrences for each object
class_counter_per_object = {}

# Dictionary to keep track of objects
objects = {}
object_id = 1

# DataFrame to store the mapping
mapping_data = []

# Y coordinate threshold for considering an object has exited
Y_EXIT_THRESHOLD = 100  # You can adjust this threshold as per your frame's dimensions



for index, row in combined_data.iterrows():
    frame, x1, y1, x2, y2, source = row['Frame'], row['X1'], row['Y1'], row['X2'], row['Y2'], row['Source']
    class_label = row['Class'] if 'Class' in row else None

    # First, deactivate objects that haven't been seen recently
    deactivate_old_objects(frame, objects, MAX_FRAMES_SINCE_LAST_SEEN, Y_EXIT_THRESHOLD)
    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2  # Calculate the center of the box
    
    distances = []
    for obj_id, obj_data in objects.items():
        if obj_data['active']:
            last_center_x, last_center_y = obj_data['last_position']
            distance = euclidean_distance(center_x, center_y, last_center_x, last_center_y)
            distances.append((distance, obj_id))
    
    # Sort distances to find the closest object
    distances.sort()

    # Try to match with the closest object within the threshold
    matched = False
    for distance, obj_id in distances:
        if distance < DISTANCE_THRESHOLD:
            obj_data = objects[obj_id]
            obj_data['last_position'] = (center_x, center_y)
            obj_data['last_frame_seen'] = frame
            if source == 'Product' and class_label:
                class_counter = class_counter_per_object.setdefault(obj_id, Counter())
                class_counter[class_label] += 1
            matched = True
            break


    # Log detailed calculation for matched or new object
    if not matched:
        objects[object_id] = {
            'first_frame_seen': frame,
            'last_frame_seen': frame,
            'last_position': (center_x, center_y),
            'active': True,
            'class': class_label if source == 'Product' else None
        }
        if source == 'Product' and class_label:
            class_counter_per_object[object_id] = Counter({class_label: 1})
        object_id += 1

# No need to save the mapping DataFrame to Excel

# Print the most frequented class for each object ID from 'Product' source
for obj_id, class_counter in class_counter_per_object.items():
    if class_counter:
        most_common_class, _ = class_counter.most_common(1)[0]
        print(f"Object ID: {obj_id}, Most Frequented Class: {most_common_class}")
