import numpy as np
import cv2 as cv
import os
import shutil   ## Allows deletion of non-empty directories

## Defines the standard video input filename
input_filename = "test2.mp4"

## Defines the file location of the facial classifier .xml file, then inits the classifier
casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml"
faceCascade = cv.CascadeClassifier(casc_path)

framerate = 10    ## Determines how long each frame is displayed on screen for
faces_found = 0
directory = "roi_frames"
if os.path.exists(directory):  ## Checks if directory exists
    shutil.rmtree(directory)   ## Deletes directory
os.mkdir(directory)            ## Creates empty directory

selection_valid = False  ## Defines user selection validator

print("Which video source should be used?")
print("0: default video file")
print("1: web cam")
while(selection_valid == False):
        selection = int(input("Please enter your selection (0/1):  "))
        if selection == 0:
            cap = cv.VideoCapture(input_filename) ## Sets capture object to standard video file
            selection_valid = True
        elif selection == 1:
            cap = cv.VideoCapture(0) ## Sets capture object to default camera input i.e. webcam
            selection_valid = True
        else:
            print("Invalid Argument")
            selection_valid = False
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

## Changes to newly created directory
os.chdir("roi_frames")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    ## converts input frame to grayscale
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)  ##detects faces within greyscale image
        for (x,y,w,h) in faces:  ## for every face detected in the greyscale image
            imageName = 'frame' + str(faces_found) + '.jpg' ## defines output filename
            faces_found += 1 ## increments face counter
            cv.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),3) ## Draw a white rectangle around that face on the input frame
            roi_gray = gray[y:y+h, x:x+w]  ## Gets the region-of-interest from the grayscale frame
            roi_colour = frame[y:y+h, x:x+w] ## Gets the region-of-interest in the colour frame
            cv.imwrite(imageName, roi_colour) ## Writes the colour region-of-interest to an image
	
        cv.imshow('Frame',frame) ## Display frame with any faces identified
        if cv.waitKey(framerate)  & 0xFF == ord('q'): ## Waits framerate period of time and allows to break out of video by pressing q
            break 
    else:
        break
 
cap.release() ## Releases the video capture object
cv.destroyAllWindows() # Closes all OpenCV windows
