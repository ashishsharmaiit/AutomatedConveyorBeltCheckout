import cv2
import os

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'BeltVideo.MOV_out_excel.MOV')

def save_frame(video_path, frame_numbers, output_folder):
    """
    Save frames from a video at given frame numbers.

    :param video_path: Path to the video file.
    :param frame_numbers: List of frame numbers to extract.
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

        # Save the frame if the current frame number is in the list
        if frame_id in frame_numbers:
            frame_file = f"{output_folder}/frame_{frame_id}.jpg"
            cv2.imwrite(frame_file, frame)
            print(f"Saved frame {frame_id} to {frame_file}")

        frame_id += 1

    cap.release()


# Define the frames you want to extract
frame_numbers = [79, 85, 101, 109, 147, 220, 888, 911]  # Replace with your specific frame numbers

# Define the output folder where the images will be saved
output_folder = os.path.join('.', 'videos')

# Call the function
save_frame(video_path, frame_numbers, output_folder)
