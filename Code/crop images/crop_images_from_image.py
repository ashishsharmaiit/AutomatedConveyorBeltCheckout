import os
import cv2
from ultralytics import YOLO

IMAGES_DIR = os.path.join('.', 'cropped_images')

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Instead of video_path, specify the path of your image
frame_path = os.path.join('.', 'IMG_7134.jpg')

# Load the image (a single frame in this case)
frame = cv2.imread(frame_path)

model_path = os.path.join('.', 'BoxModel', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.3

# No need for frame_count since we are processing a single image
results = model(frame)[0]

box_count = 1  # Initialize box counter

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Crop and save each box as a separate image
        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_img_filename = os.path.join(IMAGES_DIR, f'cropped_{box_count}.jpg')
        cv2.imwrite(cropped_img_filename, cropped_img)
        box_count += 1

        # Optionally, draw rectangles on the original frame for visualization (not saved)
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# No need to release a capture or destroy windows, as we're not using a video file
