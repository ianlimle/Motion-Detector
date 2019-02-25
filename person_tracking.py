# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 00:40:25 2019

@author: Ian
"""

from imageai.Detection import VideoObjectDetection
import os
import cv2

#obtain path to folder where the .py file runs
execution_path= os.getcwd()

camera= cv2.VideoCapture()

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")


#instantiate the VideoObjectDetection class 
detector= VideoObjectDetection()
#set model type as Retina Net
detector.setModelTypeAsRetinaNet()
##set model path to the RetinaNet model file 
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="fast")

custom_objects= detector.CustomObjects(person=True)

video_path= detector.detectObjectsFromVideo(camera_input= camera,
                                            output_file_path= os.path.join(execution_path, "person_detected"),
                                            frames_per_second=20, per_second_function=forSeconds, 
                                            per_frame_function = forFrame, per_minute_function= forMinute,
                                            log_progress=True, frame_detection_interval= 20,
                                            minimum_percentage_probability=60)

print(video_path)

