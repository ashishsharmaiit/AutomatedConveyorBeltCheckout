import os
import cv2
from ultralytics import YOLO
import random

class BoxIdentifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    @staticmethod
    def generate_color():
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def process_object_detection(self, frame):
        if frame is None:
            raise ValueError("Received an invalid frame.")

        threshold = 0.5
        detections_list = []  # Use a list to collect detections
        results = self.model(frame, verbose=False)[0]
        box_id = 0
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                class_name = results.names[int(class_id)]
                box_id += 1
                new_detection = {'BoxID': box_id, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}
                detections_list.append(new_detection)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                label = f"{class_name.upper()} {box_id}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        detection_results = {'detections': detections_list}

        return detection_results
