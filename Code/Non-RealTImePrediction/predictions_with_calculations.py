import pandas as pd
import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Load your data from both models
data1 = pd.read_excel('belt_detected_products.xlsx')
data2 = pd.read_excel('belt_detected_boxes.xlsx')

# Add a source column to each DataFrame before concatenation
data1['Source'] = 'Product'
data2['Source'] = 'Box'

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

for index, row in combined_data.iterrows():
    frame, x1, y1, x2, y2, source = row['Frame'], row['X1'], row['Y1'], row['X2'], row['Y2'], row['Source']
    class_label = row['Class'] if 'Class' in row else None
    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2  # Calculate the center of the box
    
    # Check if the object matches with any existing object
    matched = False
    match_distance = None
    for obj_id, obj_data in objects.items():
        last_center_x, last_center_y = obj_data['last_position']
        distance = euclidean_distance(center_x, center_y, last_center_x, last_center_y)
        if distance < DISTANCE_THRESHOLD:
            # Update object position and optionally the class
            objects[obj_id]['last_position'] = (center_x, center_y)
            if class_label:
                objects[obj_id]['class'] = class_label
            matched = True
            match_distance = distance
            break
    
    # Log detailed calculation for matched or new object
    if matched:
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
            'Class': objects[obj_id].get('class')
        })
    else:
        objects[object_id] = {'first_frame': frame, 'last_position': (center_x, center_y), 'class': class_label}
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
mapping_df = pd.DataFrame(mapping_data, columns=[
    'Frame', 'Object_ID', 'Source', 'Center_X', 'Center_Y', 'Last_Center_X', 'Last_Center_Y', 'Distance', 'Matched', 'Class'
])

# Save the mapping DataFrame to an Excel file
mapping_df.to_excel('detailed_object_mapping.xlsx', index=False)

# Print the results
print("Detailed object mapping saved to 'detailed_object_mapping.xlsx'.")
