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


    #Calculate the eigenValues and eigenVectors
    def __eigenVectorValue(self, matrix):

        eigenValues, eigenVectors = LA.eig(matrix)


        #Plot of energy
        fig1 = plt.figure(1)
        fig1.suptitle('Principal Components', fontsize=14, fontweight='bold')

        ax = fig1.add_subplot(1,1,1)
        ax.set_xlabel('Eigen Value', fontweight='bold')
        ax.set_ylabel('Energy', fontweight='bold')

        # BAR
        y_axis = eigenValues
        x_axis = range(len(y_axis))
        width_n = 0.4
        bar_color = 'red'

        plt.bar(x_axis, y_axis, width=width_n, color=bar_color)
        plt.show()

        return eigenValues, eigenVectors

    #Method to compare the faces with the eigenfaces
    def __compareImages(self, originalFaces, transformedFaces):
        if len(transformedFaces) != len(originalFaces):
            raise "The images sizes must be equal"

        M = len(transformedFaces)
        transformedFaceList = []
        originalFaceList = []

        for i in range(M):
            transformedFace = transformedFaces[i][:].reshape(self.M, self.N, self.O)
            transformedFace = correctMatrixValues(transformedFace)
            transformedFaceList.append(transformedFace)

            originalFace = originalFaces[i][:].reshape(self.M, self.N, self.O)
            originalFace = correctMatrixValues(originalFace)
            originalFaceList.append(originalFace)

            cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image', 1200, 600)
            imageComparison = np.hstack((originalFace, transformedFace))
            cv2.imshow('Image', imageComparison)
            cv2.waitKey(0)

    #Get the eigenfaces with a precision 0~100%.
    #With this precision I'll filter the principal components
    def getEigenFaces(self, meanFaces, precision=None):
        covarianceMatrix = self.__covarianceMatrix(meanFaces)
        eigenValues, eigenVectors = self.__eigenVectorValue(covarianceMatrix)

        if precision != None:

            if precision > 100:
                precision = 100
            elif precision < 0:
                precision = 0

            precision = precision / 100.0
            numberOfElements = int(precision * len(eigenVectors[0]))
            eigenVectors = eigenVectors[:][0:numberOfElements]

        eigenFaces = np.dot(eigenVectors, meanFaces)
        return eigenFaces

    
