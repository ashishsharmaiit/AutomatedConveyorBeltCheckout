import math
from collections import Counter

class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.object_id = 1
        self.distance_threshold = 100
        self.max_frames_since_last_seen = 40
        self.Y_EXIT_THRESHOLD = 100
        self.MIN_CONSECUTIVE_DETECTIONS = 3

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update_objects(self, detections, current_frame):
        updated_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection['X1'], detection['Y1'], detection['X2'], detection['Y2']
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            matched_obj_id = None
            for obj_id, obj_data in self.objects.items():
                if obj_data['active']:
                    last_center_x, last_center_y = obj_data['last_position']
                    distance = self.euclidean_distance(center_x, center_y, last_center_x, last_center_y)
                    if distance < self.distance_threshold:
                        obj_data['last_position'] = (center_x, center_y)
                        obj_data['last_frame_seen'] = current_frame
                        obj_data['confidence'] = obj_data.get('confidence', 0) + 1
                        if obj_data['confidence'] >= self.MIN_CONSECUTIVE_DETECTIONS:
                            matched_obj_id = obj_id
                        break

            if matched_obj_id is None:
                self.objects[self.object_id] = {
                    'last_position': (center_x, center_y),
                    'last_frame_seen': current_frame,
                    'active': True,
                    'confidence': 1
                }
                if self.objects[self.object_id]['confidence'] >= self.MIN_CONSECUTIVE_DETECTIONS:
                    matched_obj_id = self.object_id
                self.object_id += 1

            if matched_obj_id:
                updated_detection = detection.copy()
                updated_detection['ObjectID'] = matched_obj_id
                updated_detections.append(updated_detection)

        self.deactivate_old_objects(current_frame)
        return updated_detections


    def deactivate_old_objects(self, current_frame):
        for obj_id, obj_data in list(self.objects.items()):
            frames_since_seen = current_frame - obj_data['last_frame_seen']
            _, last_center_y = obj_data['last_position']
            if (frames_since_seen > self.max_frames_since_last_seen and last_center_y <= self.Y_EXIT_THRESHOLD) and obj_data['confidence'] < self.MIN_CONSECUTIVE_DETECTIONS:
                obj_data['active'] = False
