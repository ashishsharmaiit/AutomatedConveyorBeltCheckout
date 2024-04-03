from ultralytics import YOLO
import os

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

model = YOLO(model_path) 

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'BeltVideo.MOV')

results = model.track(source=video_path, show=True, tracker='bytetrack.yaml')