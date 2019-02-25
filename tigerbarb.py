# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:46:01 2019

@author: Ian
"""
import time
import math
import re
import os
import sys
import random
import numpy as np 
import cv2
import matplotlib
import matplotlib.pyplot as plt

from pyimagesearch.config import Config
import pyimagesearch.model as modellib
from pyimagesearch import utils
from pyimagesearch.model import log


"""Configuration for training on the the tigerbarb dataset.
Derives from the base config class and overrrides values specific to the tiger barb dataset
"""

class TigerBarbConfig(Config):
    
    NAME = "tigerbarb"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    #including background and tiger barb
    NUM_CLASSES = 1 + 1
    
    STEPS_PER_EPOCH = 1000
    
    VALIDATION_STEPS = 80
    
    #use smaller images for faster training
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    
    #length of square anchor side in pixels
    #use smaller anchors because the image and objects are small
    RPN_ANCHOR_SCALES = (8, 16,32, 64, 128)
    
    #reduce training ROIs per image because the images are small and
    #have few objects
    TRAIN_ROIS_PER_IMAGE = 32
    
config = TigerBarbConfig()
config.display()    

#Returns a matplotlib axes array to be used in all visualizations 
#provides a central point to control graph sizes    
def get_ax(rows=1, cols=1, size =8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class TigerBarbDataset(utils.Dataset):
    
    def load_tigerbarb(self, count, height, width):
        self.add_class("TigerBarb", 1, "TigerBarb")
        for i in range(count):
            self.add_image("TigerBarb", image_id=i, path='C:/Users/Ian/Desktop/TigerBarb', width=width, height=height)
            
    def load_mask(self, image_id):
        info= self.image_info[image_id]
        tigerbarb= info["TigerBarb"]
        count= len(tigerbarb)
        mask= np.zeros([info["height"], info["width"], count], dtype= np.uint8)
        
        occlusion= np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i]= mask[:, :, i] * occlusion
            occlusion= np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            
        class_ids= np.array([self.class_names.index(s[0]) for s in tigerbarb])
        return mask.astype(np.bool), class_ids.astype(np.uint32)
        
              