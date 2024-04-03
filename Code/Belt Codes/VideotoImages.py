import cv2
import os

# Path to your video
video_path = './VideoImagesTrials/Video6withLowSpeed.avi'

# Folder where you want to save the images
save_folder = './VideoImagesTrials/Images6'

# Check if the save folder exists, if not, create it
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
save_frame_count = 0

while True:
    # Read a frame
    ret, frame = cap.read()

    # If read was successful
    if ret:
        # Increment frame count
        frame_count += 1

        # Check if it is the 5th frame
        if frame_count % 1 == 0:
            # Save frame as image in the specified folder
            cv2.imwrite(os.path.join(save_folder, f'frame_{save_frame_count}.jpg'), frame)
            save_frame_count += 1
    else:
        # Break the loop if there are no frames left to read
        break

# Release the video capture object
cap.release()
