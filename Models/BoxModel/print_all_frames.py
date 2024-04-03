import cv2
import os

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'BeltVideo.MOV_out_excel.MOV')

def save_all_frames(video_path, output_folder):
    """
    Save all frames from a video.

    :param video_path: Path to the video file.
    :param output_folder: Folder to save the extracted frames.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if no frames are left
        if not ret:
            break

        # Save every frame
        frame_file = f"{output_folder}/frame_{frame_id}.jpg"
        cv2.imwrite(frame_file, frame)
        print(f"Saved frame {frame_id} to {frame_file}")

        frame_id += 1

    cap.release()

# Define the output folder where the images will be saved
output_folder = os.path.join('.', 'videos', 'AllFrames')

# Call the function
save_all_frames(video_path, output_folder)
