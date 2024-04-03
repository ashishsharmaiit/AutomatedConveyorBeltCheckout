import pandas as pd
import math

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
            # Update the object with the current detection
            obj_data = objects[obj_id]
            obj_data['last_position'] = (center_x, center_y)
            obj_data['last_frame_seen'] = frame
            if class_label:
                obj_data['class'] = class_label
            matched = True
            match_distance = distance
            break

    # Log detailed calculation for matched or new object
    if matched:
        # Existing object updated
        mapping_data.append({
            'Frame': frame,
            'Object_ID': obj_id,
            'Source': source,
            'Center_X': center_x,
            'Center_Y': center_y,
            'Last_Center_X': last_center_x,
            'Last_Center_Y': last_center_y,
            'Distance': match_distance,
            'Matched': 'Yes',
            'Class': obj_data.get('class')
        })
    else:
        # New object created
        objects[object_id] = {
            'first_frame': frame, 
            'last_frame_seen': frame,
            'last_position': (center_x, center_y), 
            'active': True,
            'class': class_label
        }
        mapping_data.append({
            'Frame': frame,
            'Object_ID': object_id,
            'Source': source,
            'Center_X': center_x,
            'Center_Y': center_y,
            'Last_Center_X': None,
            'Last_Center_Y': None,
            'Distance': None,
            'Matched': 'No',
            'Class': class_label
        })
        object_id += 1

# Create a DataFrame from the mapping data
mapping_df = pd.DataFrame(mapping_data)

# Save the mapping DataFrame to an Excel file
mapping_df.to_excel('detailed_object_mapping_deactivation.xlsx', index=False)

# Print the results
print("Detailed object mapping saved to 'detailed_object_mapping_deactivation.xlsx'.")
