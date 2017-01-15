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

    def __init__(self, name, images, directory):
        super(EigenPerson, self).__init__(name, images, directory)
        self.__facesMatrix = []

    def setFacesMatrix(self, channels):

        for imageName in self.getImages():
            imageUrl = self.getName()+DELIMITER+imageName
            image = readImage(self.getDirectory()+'/'+imageUrl, channels)
            self.__facesMatrix.append(image.flatten())

        # print '----------------- Tamanho da faceMatrix da pessoa:', self.getName(), ': ', np.shape(self.__facesMatrix)

    def getFacesMatrix(self):
        return self.__facesMatrix

    def getFacesMatrixDimensions(self):
        return np.shape(self.__facesMatrix)

    def getAverageFacesMatrix(self, average):
        (M, N) = self.__facesMatrix.shape
        # average = np.zeros((N), dtype=np.float32)
        for j in range(N):
            for i in range(M):
                average[j] += faces[i][j]
            average[j] = average[j] / N
