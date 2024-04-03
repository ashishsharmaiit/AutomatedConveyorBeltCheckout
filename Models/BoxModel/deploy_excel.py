import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Importing necessary libraries for Excel file handling
import openpyxl

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'BeltVideo.MOV')
video_path_out = '{}_out_excel.MOV'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Initialize DataFrame to store results
columns = ['Frame', 'X1', 'Y1', 'X2', 'Y2', 'Confidence', 'Class']
results_df = pd.DataFrame(columns=columns)

# Initialize frame count
frame_count = 0

while ret:
    frame_count += 1
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = results.names[int(class_id)]

            # Log detected boxes and class names into the DataFrame
            new_row = {
                'Frame': frame_count,
                'X1': x1,
                'Y1': y1,
                'X2': x2,
                'Y2': y2,
                'Confidence': score,
                'Class': class_name
            }
            new_row_df = pd.DataFrame([new_row])
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            # Drawing boxes and class names on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Save results to Excel file
results_df.to_excel('belt_detected_boxes.xlsx', index=False)
