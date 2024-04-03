import cv2

class VideoFrameExtractor:
    def __init__(self, video_path):
        self.video_path = video_path

    def extract_frames(self, frame_interval):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                yield frame_count, frame

            frame_count += 1

        cap.release()
