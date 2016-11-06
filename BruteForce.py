# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class BruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '
    
