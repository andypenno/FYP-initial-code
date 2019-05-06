import numpy as np
import cv2 as cv
import os
import shutil   ## Allows deletion of non-empty directories
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.ioff() ## Turns interactive plotting off (allows plt figures to be created, saved and then closed without displaying)
from scipy.signal import butter, lfilter
import csv

###############################################################################

show_video_source = False       ## Set to false to stop showing video source (will be set to True for webcam input video)
show_roi_frames = False         ## Set to false to stop showing video of ROI frames
write_roi_frames = False        ## Set to false to stop writing of ROI frames
show_initial_signal = False     ## Set to true to show initial data figure output in iPython 
write_initial_signal = False    ## Set to false to stop writing of initial data figure output
show_normalised_signal = False  ## Set to true to show normalised data figure output in iPython 
write_normalised_signal = True  ## Set to false to stop writing of normalised data figure output
show_detrended_signal = False   ## Set to true to show detrended data figure output in iPython 
write_detrended_signal = True   ## Set to false to stop writing of detrended data figure output

###############################################################################
working_directory = os.path.dirname(os.path.realpath(__file__)) ## Stores the working directory

if write_roi_frames:
    roi_directory = "roi_frames"       ## Defines the name of the directory to be used for storing of ROI frames

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
def signal_extract(facial_data,
                   r,
                   g,
                   b
                   ):
    output = []
    
    if r:
        red_frame = facial_data.raw_data[:,:,2]
        red_point = np.average(red_frame)
        output.append(red_point)
    else:
        output.append(0.00)
    
    if g:
        green_frame = facial_data.raw_data[:,:,1]
        green_point = np.average(green_frame)
        output.append(green_point)
    else:
        output.append(0.00)
    
    if b:
        blue_frame = facial_data.raw_data[:,:,0]
        blue_point = np.average(blue_frame)
        output.append(blue_point)
    else:
        output.append(0.00)
        
    return output        

###############################################################################
## Defines the standard video input filename
input_filename = "video.mp4"

## Defines the file location of the facial classifier .xml file, then inits the classifier
casc_path = "C:\ProgramData\Anaconda3\pkgs\opencv-3.4.4-py37hb76ac4c_1204\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml"
faceCascade = cv.CascadeClassifier(casc_path)

total_frames = -1
framerate = 35     ## Determines how long each frame is displayed on screen for

selection_valid = False  ## Defines user selection validator

print("Which video source should be used?")
print("0: default video file")
print("1: web cam")
while(selection_valid == False):
        selection = int(input("Please enter your selection (0/1):  "))
        if selection == 0:
            cap = cv.VideoCapture(input_filename) ## Sets capture object to standard video file
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            selection_valid = True
        elif selection == 1:
            cap = cv.VideoCapture(0) ## Sets capture object to default camera input i.e. webcam
            show_video_source = True ## Have to quit to stop recording, so needs to be shown
            selection_valid = True
        else:
            print("Invalid Argument.")
            selection_valid = False
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("\nError opening video stream or file.")
else:
    print("\nVideo input opened. Reading...")
frame_data = [] ## Initialises pointers for lists of class objects
init_facial_data = []

fps = cap.get(cv.CAP_PROP_FPS)
if fps < 1.0:
    fps = 15.0
frames_passed = 0  ## Counts how many frames ar passed
face_found = False ## Determines if a face has been found in frame
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame

    ret, frame = cap.read() ## Reads frame by frame, returning boolean and image
    if ret == True:
        if selection == 0:
            if total_frames > 100:
                if ((frames_passed % int(total_frames/100)) == 0):
                    print('Percent complete: %.2f' % (100 * frames_passed/total_frames),'%  Faces Passed - ', str(len(init_facial_data)), '/', str(frames_passed))
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
            if cv.waitKey(framerate) & 0xFF == ord('q'): ## Waits framerate period of time and allows to break out of video by pressing q
                break 
    else:
        if (frames_passed == 0):
            print("Error reading video source")
        break ## exits while loop once every frame has been passed

del gray  ## Deletes used variables to free up system memory
del img
del roi_colour

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
            if frame_counter == len(frame_data): ## Stops overflowing list
                frame_counter = len(frame_data) - 1
        else: ## Error catch
            print("Error building list of faces")
            break

print("\nFrames passed: " + str(frames_passed))
print("Faces found: " + str(len(init_facial_data)))
print("Faces estimated: " + str(len(facial_data) - len(init_facial_data)))
print("Total ROI frames: " + str(len(facial_data)))

del frame_data  ## Deletes used variables to free up system memory
del init_facial_data

if (write_roi_frames) | (show_roi_frames):
    if write_roi_frames:
        os.chdir(roi_directory) ## Changes to directory for ROI frame storage

    for i in range(0, len(facial_data)):
        if write_roi_frames:
            image_name = 'roi_frame_' + str(i) + '.jpg'     ## defines output filename
            cv.imwrite(image_name, facial_data[i].raw_data) ## Writes the roi to an image
        if show_roi_frames:
            cv.imshow('Faces', facial_data[i].raw_data)         ## Display facial data
            if cv.waitKey(framerate)  & 0xFF == ord('q'):       ## Waits framerate period of time and allows to break out of video by pressing q
                break

    if write_roi_frames:
        os.chdir(working_directory) ## Changes to directory back to working directory


## Could add more preprocessing here 

print("\nForming initial signals from list of faces...")

r_enable = True  ## Determines if a signal is formed for the R channel of the ROI frames
g_enable = True  ## Determines if a signal is formed for the G channel of the ROI frames
b_enable = True  ## Determines if a signal is formed for the B channel of the ROI frames

r_signal = []  ## Initialises lists
g_signal = []
b_signal = []

fig_width = len(facial_data)/20
fig_height = 20
   
for i in range(0, len(facial_data)):        
    components = signal_extract(facial_data[i], r_enable, g_enable, b_enable)
    r_signal.append(components[0])
    g_signal.append(components[1])
    b_signal.append(components[2])

print("Initial signals formed.")

if (write_initial_signal) | (show_initial_signal):
    fig = plt.figure(figsize=(fig_width, fig_height)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

    plt.subplot(3, 1, 1)
    plt.plot(r_signal, 'r')	## Plots red signal

    plt.subplot(3, 1, 2)
    plt.plot(g_signal, 'g')	## Plots green signal
      
    plt.subplot(3, 1, 3)
    plt.plot(b_signal, 'b')	## Plots blue signal
    if write_initial_signal:
        initial_plot_name = "plot_initial.png"
        fig.savefig(initial_plot_name) ## Saves the output figure to a .png image file
        print("Initial data figure saved to: " + initial_plot_name)
        
    if show_initial_signal:
        plt.show()
    else:
        plt.close(fig)


print("\nNormalising signals from initial signal data...")
r_normalised = []
g_normalised = []
b_normalised = []

r_signal_mean = np.average(r_signal)
r_signal_stdev = np.std(r_signal)

g_signal_mean = np.average(g_signal)
g_signal_stdev = np.std(g_signal)

b_signal_mean = np.average(b_signal)
b_signal_stdev = np.std(b_signal)

for i in range(0, len(r_signal)):
    r_normalised.append((r_signal[i] - r_signal_mean)/r_signal_stdev)
    g_normalised.append((g_signal[i] - g_signal_mean)/g_signal_stdev)
    b_normalised.append((b_signal[i] - b_signal_mean)/b_signal_stdev)    

print("Signals normalised.")

if (write_normalised_signal) | (show_normalised_signal):
    fig = plt.figure(figsize=(fig_width, fig_height)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

    plt.subplot(3, 1, 1)
    plt.plot(r_normalised, 'r')	## Plots normalised red signal

    plt.subplot(3, 1, 2)
    plt.plot(g_normalised, 'g')	## Plots normalised green signal
      
    plt.subplot(3, 1, 3)
    plt.plot(b_normalised, 'b')	## Plots normalised blue signal
    
    if write_normalised_signal:
        normalised_plot_name = "plot_normalised.png"
        fig.savefig(normalised_plot_name) ## Saves the output figure to a .png image file
        print("Normalised data figure saved to: " + normalised_plot_name)
        
    if show_normalised_signal:
        plt.show()
    else:
        plt.close(fig)


## Need to 'detrend' normalised signals
##
print("\nDetrending signals from normalised signal data...")
        
X = [i for i in range(0, len(r_normalised))]
X = np.reshape(X, (len(X), 1)) ##???

r_model = LinearRegression()
r_model.fit(X, r_normalised)

g_model = LinearRegression()
g_model.fit(X, g_normalised)

b_model = LinearRegression()
b_model.fit(X, b_normalised)
# calculate trend
r_trend = r_model.predict(X)
g_trend = r_model.predict(X)
b_trend = r_model.predict(X)
## detrend
r_detrend = [r_normalised[i] - r_trend[i] for i in range(0, len(r_normalised))]
g_detrend = [g_normalised[i] - g_trend[i] for i in range(0, len(g_normalised))]
b_detrend = [b_normalised[i] - b_trend[i] for i in range(0, len(b_normalised))]
##

print("Signals detrended.")

if (write_detrended_signal) | (show_detrended_signal):
    fig = plt.figure(figsize=(fig_width, fig_height)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

    plt.subplot(3, 1, 1)
    plt.plot(r_detrend, 'r')	## Plots normalised red signal

    plt.subplot(3, 1, 2)
    plt.plot(g_detrend, 'g')	## Plots normalised green signal
      
    plt.subplot(3, 1, 3)
    plt.plot(b_detrend, 'b')	## Plots normalised blue signal
    
    if write_detrended_signal:
        detrended_plot_name = "plot_detrended.png"
        fig.savefig(detrended_plot_name) ## Saves the output figure to a .png image file
        print("Detrended data figure saved to: " + detrended_plot_name)
        
    if show_detrended_signal:
        plt.show()
    else:
        plt.close(fig)

## Applies ICA to normalised signals
##
print("\nComputing ICA from detrended signals data...")

ica = FastICA(n_components=3, max_iter = 50000)
ica_input = list(zip(r_detrend, g_detrend, b_detrend))
ica_output = ica.fit_transform(ica_input)  # Reconstruct signals

print("ICA computed.")

s1 = ica_output[:,0]
s2 = ica_output[:,1]
s3 = ica_output[:,2]

x = [(i/fps) for i in range(0, len(s1))]

fig = plt.figure(figsize=(fig_width, fig_height)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

plt.subplot(3, 1, 1)
plt.plot(x, s1, 'k')	## Plots ica output signal 1

plt.subplot(3, 1, 2)
plt.plot(x, s2, 'k')	## Plots ica output signal 2
     
plt.subplot(3, 1, 3)
plt.plot(x, s3, 'k')	## Plots ica output signal 3
    
ica_plot_name = "plot_ica.png"
fig.savefig(ica_plot_name) ## Saves the output figure to a .png image file
print("ICA data figure saved to: " + ica_plot_name)
        
plt.close(fig)

##

## Applies FFT to ICA output
##

f1 = abs(np.fft.fft(s1))
f2 = abs(np.fft.fft(s2))
f3 = abs(np.fft.fft(s3))
freq = [((k * fps)/len(s1)) for k in range(0, len(f1))] ## Calulates frequencies

fig = plt.figure(figsize=(int(fig_width/2), fig_height)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

plt.subplot(3, 1, 1)
plt.plot(freq[0:int(len(freq)/2)], f1[0:int(len(f1)/2)], 'k')

plt.subplot(3, 1, 2)
plt.plot(freq[0:int(len(freq)/2)], f2[0:int(len(f2)/2)], 'k')
     
plt.subplot(3, 1, 3)
plt.plot(freq[0:int(len(freq)/2)], f3[0:int(len(f3)/2)], 'k')
    
fft_plot_name = "plot_fft.png"
fig.savefig(fft_plot_name) ## Saves the output figure to a .png image file
print("\nFFT data figure saved to: " + fft_plot_name)
        
plt.close(fig)

##
## By this point, one of the three output waveforms should be the rPPG signal
## Can then applying filtering techniques
## Could also use interval between successive peaks for HR

cv.destroyAllWindows() # Closes all OpenCV windows
