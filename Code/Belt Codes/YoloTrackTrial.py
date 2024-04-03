from ultralytics import YOLO
import os

model_path = os.path.join('.', 'BeltModelTraining', 'runs', 'detect', 'train5', 'weights', 'best.pt')

model = YOLO(model_path) 

VIDEOS_DIR = os.path.join('.', 'VideoImagesTrials')

video_path = os.path.join(VIDEOS_DIR, 'Video6withLowSpeed.avi')

results = model.track(source=video_path, conf=0.8, iou=0.3, persist=True, show=True, tracker='bytetrack.yaml')