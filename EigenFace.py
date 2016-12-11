# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os

from Person import *
from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '_'
AVERAGE   = 'average'

class EigenFace(object):
    'This class will extract the main components of a image using PCA '

    def __init__(self, urlTestImage, channels=0):
        self.__people = None
        self.__channels = channels
        self.__testImage = readImage(urlTestImage, self.__channels)
        self.__testImage = np.float32(self.__testImage)

        try:
            self.M, self.N, self.O = self.__testImage.shape
        except:
            self.M, self.N = self.__testImage.shape
            self.O = 1
            self.__testImage = self.__testImage.reshape((self.M,self.N,self.O))

        self.personTest = Person(directory=urlTestImage, name='unknown', images=self.__testImage)
        self.__eigenFaces = None


    def setPeople(self, people):
        self.__people = people

    def getPeople(self):
        return self.__people
#-------------------------------------------------------------------------------

    #Get the faces from database and append into a list to apply eigenfaces method
    def __getFacesMatrix(self, people):
        faces = []
        for person in people:
            images = person.getImages()
            average = 0.0
            for imageName in images:
                imageUrl = person.getName()+DELIMITER+imageName
                image = readImage(person.getDirectory()+'/'+imageUrl, self.__channels)

                faces.append(image.flatten())

        return faces

    #Make a vector with the mean of all columns
    def __averageVector(self, faces):
        (M, N) = np.shape(faces)
        average = np.zeros((N), dtype=np.float32)

        for j in range(N):
            for i in range(M):
                average[j] += faces[i][j]
            average[j] = average[j] / M

        return average

    #Remove the mean of each face
    def __removeMean(self, faces, averageVector):
        (M, N) = np.shape(faces)
        newFaceMatrix = np.zeros((M,N), dtype=np.float32)

        for i in range(M):
            for j in range(N):
                newFaceMatrix[i][j] = faces[i][j] - averageVector[j]

        return newFaceMatrix

    #Or surrogate matrix
    def __covarianceMatrix(self, faces):

        facesT = faces.transpose()

        covarianceMatrix = np.dot(faces, facesT)
        (M, N) = np.shape(covarianceMatrix)

        return covarianceMatrix
