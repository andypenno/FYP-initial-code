import numpy as np
import cv2 as cv

cascPath = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt.xml"
faceCascade = cv.CascadeClassifier(cascPath)

img = cv.imread('test1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255, 255, 0),5)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
cv.imshow('img',img)
cv.waitKey(10000)
cv.destroyAllWindows()