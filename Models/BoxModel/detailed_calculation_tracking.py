import pandas as pd
import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Load your data (assuming it's in an Excel file named 'data.xlsx')
data = pd.read_excel('belt_detected_boxes.xlsx')

# Threshold for considering objects as the same
DISTANCE_THRESHOLD = 100  # Adjust this based on your specific use case

# Dictionary to keep track of objects
objects = {}
object_id = 1

# List to store detailed calculations
detailed_data = []

for index, row in data.iterrows():
    frame, x1, y1, x2, y2 = row['Frame'], row['X1'], row['Y1'], row['X2'], row['Y2']
    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2  # Calculate center of the box
    
    # Check if the object matches with any existing object
    matched = False
    for obj_id, obj_data in objects.items():
        last_center_x, last_center_y = obj_data['last_position']
        distance = euclidean_distance(center_x, center_y, last_center_x, last_center_y)
        if distance < DISTANCE_THRESHOLD:
            objects[obj_id]['last_position'] = (center_x, center_y)
            matched = True
            detailed_data.append([frame, obj_id, center_x, center_y, last_center_x, last_center_y, distance, 'Matched'])
            break

        if not matched:
            detailed_data.append([frame, obj_id, center_x, center_y, last_center_x, last_center_y, distance, 'Not Matched'])

    # If not matched with existing objects, consider it as a new object
    if not matched:
        objects[object_id] = {'first_frame': frame, 'last_position': (center_x, center_y)}
        detailed_data.append([frame, object_id, center_x, center_y, None, None, None, 'New Object'])
        object_id += 1

# Create a DataFrame from the detailed data
detailed_df = pd.DataFrame(detailed_data, columns=['Frame', 'Object_ID', 'Center_X', 'Center_Y', 'Last_Center_X', 'Last_Center_Y', 'Distance', 'Status'])

# Save the detailed DataFrame to an Excel file
detailed_df.to_excel('detailed_tracking_data.xlsx', index=False)

# Print the results
print("Unique objects identified:")
for obj_id, obj_data in objects.items():
    print(f"Object ID: {obj_id}, First Detected Frame: {obj_data['first_frame']}")
