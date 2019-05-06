import numpy as np
import cv2 as cv

cascPath = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml"
faceCascade = cv.CascadeClassifier(cascPath)

video_capture = cv.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()