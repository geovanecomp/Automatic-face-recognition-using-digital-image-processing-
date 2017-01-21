# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Utils import *

DELIMITER = '-'

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class Person(object):
    'This class represents a generic person'

    def __init__(self, name, images, directory, channels=0):
        self.__name = name
        self.__images = images
        self.__directory = directory
        self.__channels = channels

#-------------------------------------------------------------------------------

    def setName(self, name):
        self.__name = name

    def getName(self):
        return self.__name

    def setImages(self, images):
        self.__images = images

    def getImages(self):
        return self.__images

    def addImage(self, image):
        self.__images.append(image)

    def setDirectory(self, directory):
        self.__directory = directory

    def getDirectory(self):
        return self.__directory

    def setChannels(self, channels):
        self.__channels = channels

    def getChannels(self):
        return self.__channels

#-------------------------------------------------------------------------------

    def loadImages(self, quantity=None):
        if quantity == None:
            quantity = len(self.__images)

        images = [None] * quantity

        for i in range(quantity):

            imageName = self.__images[i]
            imageUrl = self.__name+DELIMITER+imageName

            image = readImage(self.__directory+'/'+imageUrl, self.__channels)
            image = correctMatrixValues(image)

            images[i] = image

        return images

#-------------------------------------------------------------------------------

    def loadFirstImage(self):

        imageName = self.__images[0]
        imageUrl = self.__name+DELIMITER+imageName

        image = readImage(self.__directory+'/'+imageUrl, self.__channels)
        image = correctMatrixValues(image)

        return image

#-------------------------------------------------------------------------------

    def getDimensionOfImage(self):
        image = self.loadFirstImage()
        try:
            M, N, O = image.shape

        except:
            M, N = image.shape
            O = 1

        return M,N,O

#-------------------------------------------------------------------------------
