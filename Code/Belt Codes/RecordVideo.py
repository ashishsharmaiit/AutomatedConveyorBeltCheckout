import cv2

# Initialize the camera
cap = cv2.VideoCapture(2)

# Set the camera resolution to 1280x720 (High Resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set the frame rate to 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify and use the actual settings
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

# Video Writer setup with MJPEG codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, actual_fps, (actual_width, actual_height))

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame into the file 'output.mov'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' on the keyboard to exit and stop recording
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
