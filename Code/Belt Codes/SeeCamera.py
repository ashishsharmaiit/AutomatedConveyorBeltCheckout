import cv2

# Initialize the camera
# Usually, 0 is the device index for the built-in webcam. 
# If you have multiple cameras (like an external USB webcam), 
# you might need to change the index to 1 or 2.
cap = cv2.VideoCapture(2) 

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
