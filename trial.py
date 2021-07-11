import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, face = cap.read()
    # Convert to grayscale
    gray_scale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    Multifaces = faceCascade.detectMultiScale(gray_scale , 1.2, 5)
    # Draw the rectangle around each face
    for (x, y, w, h) in Multifaces:
        cv2.rectangle(face, (x, y), (x+w, y+h), (0,255, 0), 2)
    # Display
    cv2.imshow('face', face)
    # Stop the cam if exit button is pressed (esc)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()