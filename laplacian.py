# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#1º Step: Get the images and define auxiliary matrices--------------------------

#originalImage = cv2.imread('source/TrainDatabase/1.jpg', 0)
#transformedImage = cv2.imread('source/TrainDatabase/1.jpg', 0)

originalImage = cv2.imread('lua.tif', 0)
lenImgX, lenImgY = np.shape(originalImage) #Number of lines and columns
#transformedImage = cv2.imread('lua.tif', 0)
transformedImage = np.zeros((lenImgX, lenImgY), dtype=np.int)

mask = np.zeros((3,3), dtype=np.int)
lenMaskX, lenMaskY = np.shape(mask)

#-------------------------------------------------------------------------------
