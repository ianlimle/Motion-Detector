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
 
# USER-SET PARAMETERS
# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
THRESHOLD_FOR_MOVEMENT = [500, 800]

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 30

# Create capture object
cap = cv2.VideoCapture(5) # Flush the stream
cap.release()
cap = cv2.VideoCapture(0) # Then start the webcam

# Initialize frame variables
first_frame = None
next_frame = None
ls = []
fishposition = []
H = None
W = None

# Initialize display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

while True:

    # Set transient motion detected as false
    transient_movement_flag = False
    
    # Read frame
    ret, frame = cap.read()
    text = "Unoccupied"
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # If there's an error in capturing
    if not ret:
        print("CAPTURE ERROR")
        continue

    # Resize and save a greyscale version of the image
    frame = imutils.resize(frame, width = 720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    
    # loop over the contours
    for c in contours:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)
        
        # If the contour is too small, ignore it, otherwise, there's transient movement
        if cv2.contourArea(c) > THRESHOLD_FOR_MOVEMENT[0] and cv2.contourArea(c) < THRESHOLD_FOR_MOVEMENT[1]:
            transient_movement_flag = True
            
            # Draw a rectangle around big enough movements
            if len(ls) <= 19:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ls.append(cv2.boundingRect(c))
                pixel_coord = [x, y, x+w, y+h]
                fishposition.append(pixel_coord)
            
            #print(fishposition)    
            
        else:
            transient_movement_flag = False

    # The moment something moves momentarily
    if transient_movement_flag == True:
        movement_persistent_flag = True
    
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
 
    if movement_persistent_counter >0:
        movement_persistent_counter -= 1
        
    else:
        info2 = "No movement detected"
        ls = []
        cv2.putText(frame, str(info2), (520,35), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

    
    # As long as there was a recent transient movement, say a movement was detected    
    
    text = [("Time stamp", (str(time.strftime("%Y/%m/%d")) + " " + str(time.strftime("%H:%M:%S")))),
            ("Camera Location ID", "WSN"),
            ("Persistence time", str(movement_persistent_counter)),
            ("Live Fish Count", str(len(ls)))
            ]
            
    # Print the text on the screen, and display the raw and processed video feeds
    for (i, (k, v)) in enumerate(text):
        info = "{}: {}".format(k, v)
        cv2.putText(frame, info, (10, H - ((i * 20) + 400)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    
    # Convert the frame_delta to color for splicing
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

    # Splice the two video frames together to make one long horizontal one
    cv2.imshow("Motion Detector Frame", np.hstack((frame_delta, frame)))

    # Interrupt trigger by pressing q to quit the openCV program
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

# Cleanup 
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
