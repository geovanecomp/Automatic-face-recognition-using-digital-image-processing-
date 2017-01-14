# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Person import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class CorrelationPerson(Person):
    'This class represents one person with the requirements for correlation method'

    def __init__(self, name, images, directory, average=None):
        super(CorrelationPerson, self).__init__(name, images,  directory)
        self.__images = images
        self.__average = average

    def setAverage(self, average):
        self.__average = average

    def getAverage(self):
        return self.__average
