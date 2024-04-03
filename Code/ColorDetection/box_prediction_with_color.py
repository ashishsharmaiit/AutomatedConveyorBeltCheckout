import os
import cv2
import pandas as pd
from colorthief import ColorThief
from ultralytics import YOLO

# No need for the top_colors_raw function since we'll use ColorThief

# Path to your image and model
image_path = 'IMG_7134.jpg'
image_path_out = 'IMG_7134_with_boxes.jpg'
model_path = os.path.join('.', 'BoxModel', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load the image
frame = cv2.imread(image_path)

# Load the model
model = YOLO(model_path)  # load a custom model

threshold = 0.3

# Initialize DataFrame to store results
columns = ['BoxID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence', 'Class', 'Color Palette']
results_df = pd.DataFrame(columns=columns)

# Run the model on the image
results = model(frame)[0]

# Counter for unique box ID
box_id = 0

# Process detections
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        class_name = results.names[int(class_id)]
        box_id += 1  # Increment the box ID

        # Extract the image area within the bounding box
        box_area = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Save the cropped box area to a temporary file
        temp_filename = f'temp_box_{box_id}.png'
        cv2.imwrite(temp_filename, box_area)
        
        # Use ColorThief to get color palette
        color_thief = ColorThief(temp_filename)
        palette = color_thief.get_palette(color_count=10, quality=1)
        
        # Remove the temporary file
        os.remove(temp_filename)

        # Log detected boxes, class names, box ID, and color palette into the DataFrame
        new_row = {
            'BoxID': box_id,
            'X1': x1,
            'Y1': y1,
            'X2': x2,
            'Y2': y2,
            'Confidence': score,
            'Class': class_name,
            'Color Palette': palette
        }
        new_row_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        # Drawing boxes, class names, and box ID on the image
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = f"{class_name.upper()} {box_id}"
        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the image with all bounding boxes, class names, and box IDs
cv2.imwrite(image_path_out, frame)

# Save results to Excel file
results_df.to_excel('detected_boxes_with_color_palette.xlsx', index=False)
