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
        self.__directory = directory
        self.__name = name
        self.__images = images
        self.__average = average

    def setDirectory(self, directory):
        self.__directory = directory

    def getDirectory(self):
        return self.__directory

    def setName(self, name):
        self.__name = name

    def getName(self):
            return self.__name

    def setImages(self, images):
        self.__images = images

    def getImages(self):
        return self.__images

    def setAverage(self, average):
        self.__average = average

    def getAverage(self):
        return self.__average
