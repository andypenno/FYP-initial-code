import numpy as np
import cv2 as cv
import os
import shutil   ## Allows deletion of non-empty directories

###############################################################################
working_directory = os.path.dirname(os.path.realpath(__file__)) ## Stores the working directory

show_video_source = True  ## Set to false to stop showing video source (will be set to True for webcam input video)
show_roi_frames = True    ## Set to false to stop showing video of ROI frames
write_roi_frames = True   ## Set to false to stop writing of ROI frames

###############################################################################

if write_roi_frames:
    roi_directory = "roi_frames" ## Defines the name of the directory to be used for storing of ROI frames

    if os.path.exists(roi_directory):  ## Checks if directory exists in working directory
        shutil.rmtree(roi_directory)   ## Deletes directory and all contents
    
    os.mkdir(roi_directory)            ## Creates empty directory

###############################################################################
class full_frame(object):
    def __init__(self, frame_number, frame):
        self.current_frame = frame_number
        self.frame = frame
        
class facial_frame(object):
    ## Initialisation script
    def __init__(self, data, frame, x, y, w, h):
        self.raw_data = data
        self.current_frame = frame
        self.x_loc = x
        self.y_loc = y
        self.width = w
        self.height = h
        
###############################################################################
def determine_missing_face(current_image,      ## Full frame where no face was found
                           facial_data,        ## Full list of all faces found
                           current_frame,      ## Current frame number
                           face_counter_value, ## Closest face after this frame in face data
                           face_range_val = 3  ## Range of nearest faces to consider
                           ):
    
    if (face_counter_value * 2) < face_range_val:  ## For starting elements
        face_range = range(0 ,face_range_val)
    elif (int(face_counter_value) + int(face_range_val/2)+1) > len(facial_data):
        face_range = range(-face_range_val, 0)  ## For end elements
    else:  ## Standard use within list
        face_range = range(-int(face_range_val/2), int(face_range_val/2)+1)
    
    x_average = y_average = w_average = h_average = 0 ## Inits variables
    
    #####
    ## This section below simply determines the average x, y, w and h values for nearest faces
    #####
    
    for i in face_range:
        x_average += facial_data[face_counter_value+i].x_loc
        y_average += facial_data[face_counter_value+i].y_loc
        w_average += facial_data[face_counter_value+i].width
        h_average += facial_data[face_counter_value+i].height
    
    x_average = int(float(x_average/face_range_val))
    y_average = int(float(y_average/face_range_val))
    w_average = int(float(w_average/face_range_val))
    h_average = int(float(h_average/face_range_val))
    
    #####
    
    ## Extracts the estimated ROI from the full frame input
    estimated_roi = current_image[(y_average):(y_average+h_average), (x_average):(x_average+w_average)]
    ## Outputs a facial frame object of the estimated ROI    
    output = facial_frame(estimated_roi, current_frame, x_average, y_average, w_average, h_average)
    
    
    return output
    
###############################################################################
## Defines the standard video input filename
input_filename = "test2.mp4"

## Defines the file location of the facial classifier .xml file, then inits the classifier
casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml"
faceCascade = cv.CascadeClassifier(casc_path)

framerate = 35     ## Determines how long each frame is displayed on screen for

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
            show_video_source = True ## Have to quit to stop recording, so needs to be shown
            selection_valid = True
        else:
            print("Invalid Argument")
            selection_valid = False
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("\nError opening video stream or file.")
else:
    print("\nVideo input opened. Reading...")
frame_data = [] ## Initialises pointers for lists of class objects
init_facial_data = []

frames_passed = 0  ## Counts how many frames ar passed
face_found = False ## Determines if a face has been found in frame
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame

    ret, frame = cap.read() ## Reads frame by frame, returning boolean and image
    if ret == True:
        img = np.array(frame) ## Copies the frame so any changes made before drawing do not affect input data
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    ## converts input frame to grayscale
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)  ## detects faces within greyscale image
        for (x,y,w,h) in faces:  ## for every face detected in the greyscale image
            face_found = True
            roi_colour = frame[y:y+h, x:x+w] ## Gets the region-of-interest in the colour frame
            init_facial_data.append(facial_frame(roi_colour, frames_passed, x, y, w, h))  ## Adds the facial region, frame number, and locators on the original image
            cv.rectangle(img,(x,y),(x+w,y+h),(255, 255, 255),3) ## Draw a white rectangle around that face on the input frame
        if face_found == False: ## If no face has been found in this frame
            frame_data.append(full_frame(frames_passed, frame))
        frames_passed += 1
        face_found = False ## Resets variable for next pass
        if show_video_source:
            cv.imshow('Overlayed Video Source',img) ## Display frame with any faces identified
            if cv.waitKey(framerate)  & 0xFF == ord('q'): ## Waits framerate period of time and allows to break out of video by pressing q
                break 
    else:
        if (frames_passed == 0):
            print("Error reading video source")
        break ## exits while loop once every frame has been passed

if (frames_passed > 0):
    print("Video finished reading. Building ROI Dataset...")

cap.release() ## Releases the video capture object

facial_data = [] ## Initialises pointer for final list of facial data objects

if len(frame_data) < 1: ## If faces have been identified for all frames
    for i in range(0, len(init_facial_data)): ## For every face, add to facial_data
        facial_data.append(facial_frame(init_facial_data[i].raw_data, init_facial_data[i].current_frame, init_facial_data[i].x_loc, init_facial_data[i].y_loc, init_facial_data[i].width, init_facial_data[i].height))
else: ## Not all frames have a face identified within
    print("Some frames missing an identified face. Estimating missing faces...\n")
    face_counter = 0
    frame_counter = 0
    for i in range(0, frames_passed):
        if (init_facial_data[face_counter].current_frame == i) & (frame_data[frame_counter].current_frame != i):
            #print("Face identified:  " + str(i) + "/" + str(frames_passed))
            facial_data.append(facial_frame(init_facial_data[face_counter].raw_data, init_facial_data[face_counter].current_frame, init_facial_data[face_counter].x_loc, init_facial_data[face_counter].y_loc, init_facial_data[face_counter].width, init_facial_data[face_counter].height))
            
            face_counter += 1
            if face_counter == len(init_facial_data): ## Stops overflowing list
                face_counter = len(init_facial_data) - 1
        elif (frame_data[frame_counter].current_frame == i) & (init_facial_data[face_counter].current_frame != i):
            print("Face estimated:   " + str(i) + "/" + str(frames_passed))
            facial_data.append(determine_missing_face(frame_data[frame_counter].frame, init_facial_data, frame_data[frame_counter].current_frame, face_counter))
                       
            frame_counter += 1
            if frame_counter == len(frame_data): ##Stops overflowing list
                frame_counter = len(frame_data) - 1
        else: ## Error catch
            print("Error building list of faces")
            break


print("\nFrames passed: " + str(frames_passed))
print("Faces found: " + str(len(init_facial_data)))
print("Faces estimated: " + str(len(facial_data) - len(init_facial_data)))
print("Faces collected in total: " + str(len(facial_data)))

## Changes to directory for ROI frame storage
os.chdir(roi_directory)

for i in range(0, len(facial_data)):
    if write_roi_frames:
        image_name = 'roi_frame_' + str(i) + '.jpg'     ## defines output filename
        cv.imwrite(image_name, facial_data[i].raw_data) ## Writes the roi to an image
    if show_roi_frames:
        cv.imshow('Faces', facial_data[i].raw_data)         ## Display facial data
        if cv.waitKey(framerate)  & 0xFF == ord('q'):       ## Waits framerate period of time and allows to break out of video by pressing q
            break

## Changes to directory back to working directory
os.chdir(working_directory)
        
cv.destroyAllWindows() # Closes all OpenCV windows
