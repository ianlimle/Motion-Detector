# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:53:30 2018

@author: Ian
"""

#Description:
#This script runs a motion detector to test proof the concept of video analytics in optimizing situational awareness of key water installations
#The aim is to contribute to a Smart Water Security Management ecosystem with artificial intelligence and automation 
#It detects transient motion in a location and if said movement is large enough, and recent enough, reports that there is motion
#This video analytics module can be interfaced to other devices or systems (ie drones, UAV or autonomous vehicles) where operational requirements demand

import imutils
import cv2
import numpy as np
import time
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
args = vars(ap.parse_args())

if not args.get("input", False):
	print("starting video stream...")
	cap = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("opening video file...")
	cap = cv2.VideoCapture(args["input"])
    
# USER-SET PARAMETERS
# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
THRESHOLD_FOR_MOVEMENT = [800, 1000]

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 30

# Create capture object
#cap = cv2.VideoCapture(5) # Flush the stream
#cap.release()
#cap = cv2.VideoCapture(0) # Then start the webcam

# Initialize frame variables
first_frame = None
next_frame = None
ls = []
fishposition = []
H = None
W = None

# instantiate our centroid tracker
# initialize a list ( ie. trackers) to store each of our dlib correlation trackers
# followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# Initialize display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0 

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalVert= 0
totalHoriz = 0

fps = FPS().start()

#loop over frames from the video stream
while True:

    # Set transient motion detected as false
    transient_movement_flag = False
    
    # Read frame
    frame = cap.read()
    frame = frame[1] if args.get("input", False) else frame
    
    #if we are viewing a video and we did not grab a frame 
    #then we have reached the end of the video 
    if args["input"] is not None and frame is None:
        break

    # Resize and save a greyscale and RGB (for dlib) version of the image 
    frame = imutils.resize(frame, width = 720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # Blur it to remove camera noise (reducing false positives)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: 
        first_frame = gray    

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appropriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

        
    # Set the next frame to compare (the current frame)
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv2.dilate(thresh, None, iterations = 2)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    
    # loop over the contours
    for c in contours:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)
        
        # If the contour is too small, ignore it, otherwise, there's transient movement
        if cv2.contourArea(c) > THRESHOLD_FOR_MOVEMENT[0] and cv2.contourArea(c) < THRESHOLD_FOR_MOVEMENT[1]:
            transient_movement_flag = True

            # Draw a rectangle around big enough movements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ls.append(cv2.boundingRect(c))
            #pixel_coord = [x, y, x+w, y+h]
            #fishposition.append(pixel_coord)
                
            objects = ct.update(ls) 
     
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                #check to see if a trackable object exists for the current
	            #object ID
                to = trackableObjects.get(objectID, None)

            	# if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

    		      # otherwise, there is a trackable object so we can utilize it
                  # to determine direction
                else:
			         #the difference between the y-coordinate of the *current*
			         # centroid and the mean of *previous* centroids will tell
			         # us in which direction the object is moving (negative for
			         # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    x = [c[0] for c in to.centroids]
                    directionY = centroid[1] - np.mean(y)
                    directionX = centroid[0] - np.mean(x)
                    to.centroids.append(centroid)
            
                    if not to.counted:
                        #if the direction is postive (indicating the object is moving down)
                        #and the centroid is below the upper horizontal line 
                        if directionY > 0 and centroid[1] > H//4 and centroid[1] < (7*H)//8:
                            totalVert += 1
                            to.counted = True
                
                        if directionY < 0 and centroid[1] < (7*H)//8 and centroid[1] > H//4:
                            totalVert += 1
                            to.counted = True
            
                        #if direction is positive (indicating the object is moving to left)
                        if directionX > 0 and centroid[0] > W//8 and centroid[0] < (7*W)//8:
                            totalHoriz += 1
                            to.counted = True
                
                        if directionX < 0 and centroid[0] < (7*W)//8 and centroid[0] > W//8:
                            totalHoriz += 1
                            to.counted = True
                
                trackableObjects[objectID] = to
                id_ = "ID {}".format(objectID)
                cv2.putText(frame, id_, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)  
            
            ls =[]
            
    #initialize fishcount variable as a count of the horizontal, vertical movements           
    fishcount = totalVert + totalHoriz
            
    #if there is movement, reset the persistent movement timer 
    if transient_movement_flag == True:
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
    
    #as long as there was a recent transient movement, 
    #the persistent movement timer will count down to zero             
    if movement_persistent_counter >0:
        info2 = "Movement detected"
        movement_persistent_counter -=1
    #before delaring there is no movement     
    elif movement_persistent_counter == 0:
        info2 = "No movement detected"
        cv2.putText(frame, str(info2), (520,35), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #reset the fishcount counter
        fishcount = 0
        totalVert = 0
        totalHoriz = 0
        
    #draw the fishtank boundaries      
    cv2.rectangle(frame, (W//8, H//4), ((7*W)//8, (7*H)//8), (255, 0, 0), 2)  
      
    text = [("Camera Location ID", "WSN"),
            ("Persistence time", str(movement_persistent_counter)),
            ("Vertical", totalVert),
            ("Horizontal", totalHoriz),
            ("Live Fish Count", str(fishcount) + " as at " + str(time.strftime("%H:%M:%S")))
            ]
            
    # Print the text on the screen, and display the raw and processed video feeds
    for (i, (k, v)) in enumerate(text):
        info = f"{k}: {v}"
        cv2.putText(frame, info, (10, H - ((i * 20) + 440)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Convert the frame_delta to color for splicing
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

    # Splice the two video frames together to make one long horizontal one
    cv2.imshow("FishCount Event", np.hstack((frame_delta, frame)))

    # Interrupt trigger by pressing q to quit the openCV program
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
    
    totalFrames += 1
    fps.update()
    fps.stop()

# Cleanup 
cv2.waitKey(0)

if not args.get("input", False):
    cap.stop()
    
else:
    cap.release()    
    
cv2.destroyAllWindows()
