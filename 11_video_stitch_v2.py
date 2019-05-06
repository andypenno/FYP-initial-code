import numpy as np
import os
import cv2 as cv
import imutils
import time

root_directory = os.path.dirname(os.path.realpath(__file__)) ## Sets the root directory to search through as current working directory
video_files_directory = "C:\\Users\\Andrew_Laptop\\Documents\\University\\Year 4\\EE40150 - FYP\\Dataset\\P001\\v1"
video_files = []

for root, dirs, files in os.walk(video_files_directory):
    for file in files:
        if file.endswith(".MP4"):
             video_files.append([root, file])
os.chdir(root_directory) ## Ensures back in working directory at end of walk

print("Video Files Identified:\n")

for i in range(len(video_files)):
    if video_files[i][1].find("A") != -1:
        print(video_files[i][1])
        A_source = os.path.join(video_files[i][0], video_files[i][1])
    elif video_files[i][1].find("B") != -1:
        print(video_files[i][1])
        B_source = os.path.join(video_files[i][0], video_files[i][1])
    elif video_files[i][1].find("C") != -1:
        print(video_files[i][1])
        C_source = os.path.join(video_files[i][0], video_files[i][1])
    elif video_files[i][1].find("D") != -1:
        print(video_files[i][1])
        D_source = os.path.join(video_files[i][0], video_files[i][1])

capA = cv.VideoCapture(A_source)
capB = cv.VideoCapture(B_source)
capC = cv.VideoCapture(C_source)
capD = cv.VideoCapture(D_source)

input_width = capB.get(cv.CAP_PROP_FRAME_WIDTH)  # 3 is video width
input_height = capB.get(cv.CAP_PROP_FRAME_HEIGHT) # 4 is video height
input_fps = capB.get(cv.CAP_PROP_FPS)    # 5 is input framerate
total_frames = capB.get(cv.CAP_PROP_FRAME_COUNT) # 7 is total frames

output_width = int(input_width * 2)
output_height = int(input_height * 2)


fourcc = cv.VideoWriter_fourcc(*'MP4V')
output_filename = os.path.join(video_files_directory, 'video.mp4')
output = cv.VideoWriter(output_filename, fourcc, input_fps, (output_width, output_height))

frames_passed = 0

print("\nVideo sources opened, stitching video sources...")

start = time.time()
            
stitched_successfully = 0    
concatenated = 0     
            
while((capA.isOpened()) & capB.isOpened() & capC.isOpened() & capD.isOpened()):
    retA, frameA = capA.read()
    retB, frameB = capB.read()
    retC, frameC = capC.read()
    retD, frameD = capD.read()
    
    frames_passed += 1
    
    if ((frames_passed % int(total_frames/100)) == 0):
                    current = time.time()
                    print('Percent complete: %.2f' % (100 * frames_passed/total_frames),'% ;  ', "Current run-time: %.2f" % (current-start), "seconds")
        
    if ((retA == True) & (retB == True) & (retC == True) & (retD == True)):
        images = [frameA, frameB, frameC, frameD]
        stitcher = cv.createStitcher(try_use_gpu = False)
        ret, stitched = stitcher.stitch(images)
        if ret == '0':
            stitched_successfully += 1
            output.write(stitched)
#            cv.imshow('Stitched Sources', stitched)  ## Uncomment to show output stitched frame
#            cv.waitKey(1) 
        else:
            concatenated += 1
            top_half = np.concatenate((frameA, frameB),axis=1)
            bottom_half = np.concatenate((frameC, frameD),axis=1)
            out_frame = np.concatenate((top_half, bottom_half), axis=0)
            output.write(out_frame)
#            cv.imshow('Combined Sources', out_frame)  ## Uncomment to show output combined frame
#            cv.waitKey(1)             
    else:
        if frames_passed == 0:
            print ("Error occured opening video files")
        break
      
capA.release()
capB.release()
capC.release()
capD.release()

output.release()

end = time.time()
print("\nStitching Complete. Saved to:  " + output_filename)
print("Percentage stitched successfully:  " + str(100 * (stitched_successfully/total_frames)) + "%")
print("Percentage concatenated:  " + str(100 * (concatenated/total_frames)) + "%")
print("Run-time: %.2f" % (end-start), "seconds")
cv.destroyAllWindows() # Closes all OpenCV windows
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        