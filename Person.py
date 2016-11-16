# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class Person(object):
    'This class represents one person'

    def __init__(self, directory, name, images, average=None):
        self.directory = directory
        self.name = name
        self.images = images
        self.average = average
 
