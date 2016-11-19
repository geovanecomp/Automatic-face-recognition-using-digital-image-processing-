# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Person import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')
URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '_'
AVERAGE   = 'average'

class CompleteBruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, urlTestImage):
        self.testImage = cv2.imread(urlTestImage)
        self.testImage = np.float32(self.testImage)
        avg = self.__averageImage(self.testImage)
        self.personTest = Person(directory=urlTestImage, name='unknown', images=self.testImage, average=avg)
        self.M, self.N, self.O = self.testImage.shape
