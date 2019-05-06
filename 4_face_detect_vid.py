import numpy as np
import cv2 as cv
import os
default_casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
alt_casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt.xml"
alt2_casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml"
input_filename = "test2.mp4"
framerate = 10  # Determines how long each frame is displayed on screen for
selection_valid = False
selection_valid2 = False
face_in_frame = False
faces_in_frames = 0
face_counter = 0
faces_found = 0
total_frames = 0


print("Which Classifier should be used?")
print("0: default")
print("1: alternate 1")
print("2: alternate 2")

while(selection_valid == False):
        selection = int(input("Please enter your selection (0-2):  "))
                
        if selection == 0:
            faceCascade = cv.CascadeClassifier(default_casc_path)
            selection_valid = True
        elif selection == 1:
            faceCascade = cv.CascadeClassifier(alt_casc_path)
            selection_valid = True
        elif selection == 2:
            faceCascade = cv.CascadeClassifier(alt2_casc_path)
            selection_valid = True
        else:
            print("Invalid Argument")
            selection_valid = False

print("Which video source should be used?")
print("0: default video file")
print("1: web cam")
while(selection_valid2 == False):
        selection = int(input("Please enter your selection (0/1):  "))
                
        if selection == 0:
            cap = cv.VideoCapture(input_filename)
            selection_valid2 = True
        elif selection == 1:
            cap = cv.VideoCapture(0)
            selection_valid2 = True
        else:
            print("Invalid Argument")
            selection_valid2 = False

 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        total_frames += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            if face_in_frame == False: ## If face is first found in frame
                faces_in_frames += 1
                face_in_frame = True
            face_counter += 1
            cv.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 0),5)
            roi_gray = gray[y:y+h, x:x+w]
            roi_colour = frame[y:y+h, x:x+w]
            
        cv.imshow('Overlayed Video Source',frame)
        face_in_frame = False
        if cv.waitKey(framerate)  & 0xFF == ord('q'): #Allows to break out of video by pressing q
            break 
    else:
        break

# Write some Text
output_text = "Facial Identification %: " + str(float(100*faces_in_frames/total_frames))
print(output_text)
output_text2 = "False Positives %: " + str(float(100*((face_counter - faces_in_frames)/total_frames)))
print(output_text2)

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()
