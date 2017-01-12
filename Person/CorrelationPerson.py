# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class CorrelationPerson(object):
    'This class represents one person with the requirements for correlation method'

    def __init__(self, directory, name, images, average=None):
        super(CorrelationPerson, self).__init__(directory, name)
        self.__images = images
        self.__average = average

    def setImages(self, images):
        self.__images = images

    def getImages(self):
        return self.__images

    def addImage(self, image):
        self.__images.append(image)

    def setAverage(self, average):
        self.__average = average

    def getAverage(self):
        return self.__average
