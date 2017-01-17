# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Person import *
from Utils import *

DELIMITER = '-'

class EigenPerson(Person):
    'This class represents one person with the requirements to eigenface method'

    def __init__(self, name, images, directory, channels=0):
        super(EigenPerson, self).__init__(name, images, directory, channels)
        self.__facesMatrix = []

    def setFacesMatrix(self):
        for imageName in self.getImages():
            imageUrl = self.getName()+DELIMITER+imageName
            image = readImage(self.getDirectory()+'/'+imageUrl, self.getChannels())
            self.__facesMatrix.append(image.flatten())


    def getFacesMatrix(self):
        return self.__facesMatrix

    def getFacesMatrixDimensions(self):
        return np.shape(self.__facesMatrix)
