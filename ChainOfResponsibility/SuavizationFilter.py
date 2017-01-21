# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class SuavizationFilter(object):
    'This class is responsible for reduce noises from an image'

    def __init__(self, nextImageProcessing=None, dimensionMask=3):
        self.__nextImageProcessing = nextImageProcessing
        self.__dimensionMask = dimensionMask

#-------------------------------------------------------------------------------

    #To remove the noises a lot of methods can be used, like average, median and others.
    #Here I'm using the median with the default dimension = 3, but we could use others.
    def averageMask(self, image, x, y, z):

        #A mask must be square
        totalElements = self.__dimensionMask * self.__dimensionMask

        vector = np.zeros(totalElements, dtype=np.float32)
        average = 0

        #Transforming a matrix in an array
        for i in range(x, x+self.__dimensionMask):
            for j in range(y, y+self.__dimensionMask):
                average += image[i][j][z]

        return average / totalElements

#-------------------------------------------------------------------------------

    def calculate(self, image):

        image = addRows(image, self.__dimensionMask - 1, 0)
        image = addRows(image, self.__dimensionMask - 1, len(image))

        image = addColumns(image, self.__dimensionMask - 1, 0)
        image = addColumns(image, self.__dimensionMask - 1, len(image[0]))

        transformedImage = np.zeros(image.shape, dtype=np.float32)

        value = 0

        lenX, lenY, lenZ = image.shape

        for x in range(lenX - (self.__dimensionMask - 1)):
            for y in range(lenY - (self.__dimensionMask - 1)):
                for z in range(lenZ):
                    value = self.averageMask(image, x, y, z)

                    if value < 0:
                        value = 0

                    if value > 255:
                        value = 255

                    transformedImage[x][y][z] = value
                    value = 0

        transformedImage = deleteRows(transformedImage, self.__dimensionMask - 1, 0)
        transformedImage = deleteRows(transformedImage, self.__dimensionMask - 1, len(transformedImage))

        transformedImage = deleteColumns(transformedImage, self.__dimensionMask - 1, 0)
        transformedImage = deleteColumns(transformedImage, self.__dimensionMask - 1, len(transformedImage[0]))

        if self.__nextImageProcessing != None:
            transformedImage = self.__nextImageProcessing.calculate(transformedImage)

        return transformedImage

#-------------------------------------------------------------------------------
