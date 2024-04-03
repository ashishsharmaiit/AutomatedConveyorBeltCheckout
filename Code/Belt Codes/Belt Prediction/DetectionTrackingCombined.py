from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])

    def process_frame(self, frame):
        results = self.model.track(frame, conf=0.8, iou=0.3, persist=True, verbose=False)
        detections_list = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Assuming that results are already in (x1, y1, x2, y2) format
            boxes = results[0].boxes.xyxy.cpu().tolist()  # Use xyxy for (x1, y1, x2, y2)
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except AttributeError:
                track_ids = list(range(len(boxes)))

            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = box

                new_detection = {'TrackID': track_id, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}
                detections_list.append(new_detection)

            self._update_track_history(track_ids, boxes)

        detection_results = {'detections': detections_list}
        return detection_results

    def _update_track_history(self, track_ids, boxes):
        for track_id, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box
            # Use center of the box for tracking history
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            track = self.track_history[track_id]
            track.append((cx, cy))
            if len(track) > 30:
                track.pop(0)
