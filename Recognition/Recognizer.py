# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from numpy import linalg as LA
from abc import ABCMeta, abstractmethod
from Person import *
from Utils import *


#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '_'
AVERAGE   = 'average'

class Recognizer(object):
    'This class will provide the basis for all recognitions'

    def __init__(self, channels=0):
        self.__people = None
        self.__channels = channels
        self.__eigenFaces = None

    def setDimensions(M, N, O=1):
        self.M = M
        self.N = N
        self.O = O

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
                self.M, self.N, self.O = image.shape
                faces.append(image.flatten())

        return faces

#-------------------------------------------------------------------------------

    #Select (and remove) randomly a number of faces from train database to test
    def getRandomFacesToTest(self, trainFaces, numberOfFaces=1):
        numberOfFaces = 5
        (M, N) = np.shape(trainFaces)
        facesToTest = np.zeros((numberOfFaces,N), dtype=np.float32)

        #Setting some faces to analyse results (Temporary)
        # facesToTest[0][:] = trainFaces[0][:]
        # trainFaces = np.delete(trainFaces, 0, 0)
        #
        # facesToTest[1][:] = trainFaces[5][:]
        # trainFaces = np.delete(trainFaces, 5, 0)
        #
        # facesToTest[2][:] = trainFaces[10][:]
        # trainFaces = np.delete(trainFaces, 10, 0)
        #
        # facesToTest[3][:] = trainFaces[15][:]
        # trainFaces = np.delete(trainFaces, 15, 0)
        #
        # facesToTest[4][:] = trainFaces[20][:]
        # trainFaces = np.delete(trainFaces, 20, 0)


        for i in range(numberOfFaces):

            #The total size of trainFaces is inversely proportional to facesToTest
            randomPosition = int(random.random()*(M-len(facesToTest)))

            facesToTest[i][:] = trainFaces[randomPosition][:]

            #0 is for the axis 0 (row)
            trainFaces = np.delete(trainFaces, randomPosition, 0)

            #One face was removed from the matrix, so his length decrease too
            numberOfFaces -= 1

        print 'Dimensions of train and test matrix', np.shape(trainFaces), np.shape(facesToTest)
        return trainFaces, facesToTest

#-------------------------------------------------------------------------------

    #Make a vector with the mean of all columns
    def __averageVector(self, faces):
        (M, N) = np.shape(faces)
        average = np.zeros((N), dtype=np.float32)

        for j in range(N):
            for i in range(M):
                average[j] += faces[i][j]
            average[j] = average[j] / N

        return average

#-------------------------------------------------------------------------------

    #Remove the mean of each face
    def __removeMean(self, faces, averageVector):
        (M, N) = np.shape(faces)
        newFaceMatrix = np.zeros((M,N), dtype=np.float32)

        for i in range(M):
            for j in range(N):
                newFaceMatrix[i][j] = faces[i][j] - averageVector[j]

        return newFaceMatrix

#-------------------------------------------------------------------------------

    #Or surrogate matrix
    def __covarianceMatrix(self, faces):

        facesT = faces.transpose()

        covarianceMatrix = np.dot(faces, facesT)
        (M, N) = np.shape(covarianceMatrix)

        return covarianceMatrix

#-------------------------------------------------------------------------------

    #Calculate the eigenValues and eigenVectors
    def __eigenVectorValue(self, matrix):

        eigenValues, eigenVectors = LA.eig(matrix)

        # Plot of energy
        # fig1 = plt.figure(1)
        # fig1.suptitle('Principal Components', fontsize=14, fontweight='bold')
        #
        # ax = fig1.add_subplot(1,1,1)
        # ax.set_xlabel('Eigen Value', fontweight='bold')
        # ax.set_ylabel('Energy', fontweight='bold')
        #
        # # BAR
        # y_axis = eigenValues
        # x_axis = range(len(y_axis))
        # width_n = 0.4
        # bar_color = 'red'
        #
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color)
        # plt.show()

        return eigenValues, eigenVectors

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

    #With the minimum euclidian distance, I'll find the most closer person
    def __euclideanDistance(self, v1, v2):

        diffSquare = 0
        for i in range(len(v1)):
                diffSquare += (v1[i] - v2[i]) ** 2

        euclideanDistance = diffSquare ** 0.5
        return euclideanDistance
        #return LA.norm(v1-v2)**2


    def __applyEuclidianDistance(self, trainFaces, testFaces):

        #The matrices must have same number of columns
        numTestFaces = len(testFaces)
        M,N = trainFaces.shape
        euclideanDistance = np.zeros((numTestFaces, M), dtype=np.float32)

        for i in range(numTestFaces):
            for x in range(M):
                    euclideanDistance[i][x] = self.__euclideanDistance(trainFaces[x][:], testFaces[i][:])

        return euclideanDistance

#-------------------------------------------------------------------------------

    def showResults(self, minError, originalFace, transformedFace):
        print "The min error was: ", minError
        print 'The person found was... '

        transformedFace = transformedFace.reshape(self.M, self.N, self.O)
        transformedFace = correctMatrixValues(transformedFace)

        originalFace = originalFace.reshape(self.M, self.N, self.O)
        originalFace = correctMatrixValues(originalFace)

        cv2.namedWindow('Faces: TrainPerson x TestPerson',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Faces: TrainPerson x TestPerson', 1200, 600)
        imageComparison = np.hstack((originalFace, transformedFace))
        cv2.imshow('Faces: TrainPerson x TestPerson', imageComparison)
        cv2.waitKey(0)

#-------------------------------------------------------------------------------

    #The main method
    def eigenFaceMethod(self, quantityPeopleToTest=1, precision=100, showResults=False):

        trainFaces = self.__getFacesMatrix(self.__people)
        trainFaces, testFaces = self.getRandomFacesToTest(trainFaces, quantityPeopleToTest)

        if self.__eigenFaces == None:

            averageVector = self.__averageVector(trainFaces)

            meanFaces = self.__removeMean(trainFaces, averageVector)

            self.__eigenFaces = self.getEigenFaces(meanFaces, precision)

        eigenTrainFaces = np.dot(self.__eigenFaces, meanFaces.transpose()) # 17x20
        eigenTrainFaces = eigenTrainFaces.transpose()

        #Applying the same operation on testImage
        # testFaces = []

        # testFaces.append(self.__testImage.flatten())

        meanTestFace = self.__removeMean(testFaces, averageVector)

        eigenTestFaces = np.dot(self.__eigenFaces, meanTestFace.transpose())
        eigenTestFaces = eigenTestFaces.transpose()

        euclideanDistances = self.__applyEuclidianDistance(eigenTrainFaces, eigenTestFaces)
        transformedFaces = []

        for i in range(len(euclideanDistances)):
            posMinValue = np.argmin(euclideanDistances[i][:])

            #Comparing the testImage X foundPerson
            transformedFace = trainFaces[posMinValue][:]
            originalFace = testFaces[i][:]

            if showResults == True:
                self.showResults(euclideanDistances[i][posMinValue], originalFace, transformedFace)

            transformedFaces.append(transformedFace)

        return transformedFaces

    #-------------------------------------------------------------------------------
