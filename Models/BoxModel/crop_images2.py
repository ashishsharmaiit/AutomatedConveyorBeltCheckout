import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
IMAGES_DIR = os.path.join('.', 'cropped_images')

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

video_path = os.path.join(VIDEOS_DIR, 'products.MOV')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

frame_count = 1  # Start frame counter with 1

while ret:

    results = model(frame)[0]

    box_count = 1  # Initialize box counter for each frame

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Crop and save each box as a separate image before drawing the rectangles
            cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
            cropped_img_filename = os.path.join(IMAGES_DIR, f'{frame_count}.{box_count}.jpg')
            cv2.imwrite(cropped_img_filename, cropped_img)
            box_count += 1

            # Optionally, draw rectangles on the original frame for visualization (not saved)
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    ret, frame = cap.read()
    frame_count += 1  # Increment frame counter

cap.release()
cv2.destroyAllWindows()
