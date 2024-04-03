import pandas as pd
import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Load your data from both models (assuming they are named 'model1.xlsx' and 'model2.xlsx')
data1 = pd.read_excel('belt_detected_products.xlsx')
data2 = pd.read_excel('belt_detected_boxes.xlsx')

# Combine the data from both models into one DataFrame
combined_data = pd.concat([data1, data2])

# Sort the combined data by the 'Frame' column to process the frames in order
combined_data.sort_values(by='Frame', inplace=True)

# Threshold for considering objects as the same
DISTANCE_THRESHOLD = 100  # Adjust this based on your specific use case

# Dictionary to keep track of objects
objects = {}
object_id = 1

for index, row in combined_data.iterrows():
    frame, x1, y1, x2, y2 = row['Frame'], row['X1'], row['Y1'], row['X2'], row['Y2']
    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2  # Calculate center of the box
    
    # Check if the object matches with any existing object
    matched = False
    for obj_id, obj_data in objects.items():
        last_center_x, last_center_y = obj_data['last_position']
        if euclidean_distance(center_x, center_y, last_center_x, last_center_y) < DISTANCE_THRESHOLD:
            objects[obj_id]['last_position'] = (center_x, center_y)
            matched = True
            break
    
    # If not matched with existing objects, consider it as a new object
    if not matched:
        objects[object_id] = {'first_frame': frame, 'last_position': (center_x, center_y)}
        object_id += 1

# Print the results
print("Unique objects identified:")
for obj_id, obj_data in objects.items():
    print(f"Object ID: {obj_id}, First Detected Frame: {obj_data['first_frame']}")
