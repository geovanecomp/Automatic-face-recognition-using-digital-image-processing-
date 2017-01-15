# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os

from Recognizer import *
from Person.EigenPerson import *
from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
# URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
# URLTRAIN    = 'Source/CompactFEI_80x60/TrainDatabase/'
URLTRAIN    = 'Source/CompactFEI_320x240/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '-'
AVERAGE   = 'average'

class EigenFace(Recognizer):
    'This class will extract the main components of a image using PCA '

    def __init__(self, quantityPeopleToTrain=None, channels=0):
        super(EigenFace, self).__init__(channels)
        self.people = self.getPeople(quantityPeopleToTrain)
        self.M, self.N, self.O = self.setDimensionsOfImage(self.people)
        self.__eigenFaces = None

        #I'll use a dictionary to map people to your faces on matrix of faces
        self.__peopleMap = {}

    def setPeople(self, people):
        self.people = people

#-------------------------------------------------------------------------------

    def getPeople(self, numberOfPeople=None):
        if numberOfPeople == None:
            #-1 its because this function count the TrainDatabase too
            numberOfPeople = len(list(os.walk(URLTRAIN))) - 1;

        people = [None] * numberOfPeople

        for i in range(numberOfPeople):

            #Getting the url, folders and files
            directory, folders, files = os.walk(URLTRAIN+str(i+1)).next()

            images = [None] * len(files)

            for (j, file) in enumerate(files):
                name, image = file.split(DELIMITER)
                images[j] = image

            person = EigenPerson(name=name, images=images, directory=directory)
            people[i] = person

        return people
#-------------------------------------------------------------------------------

    #Get the trainFaces from database and append into a list to apply eigenfaces
    # method under her faces
    def __preparePeople(self, people):
        for person in people:
            person.setFacesMatrix(self.channels)

        return people

#-------------------------------------------------------------------------------

    #Select (and remove) randomly a number of faces from train database to test
    def __getRandomPeopleToTest(self, trainPeople, numberOfPeople=1):
        M = len(trainPeople)
        # (M, N) = np.shape(trainFaces)
        # facesToTest = np.zeros((numberOfFaces,N), dtype=np.float32)

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

        # testPeople = [None] * numberOfPeople
        testPeople = []
        temporaryPerson = None

        for i in range(numberOfPeople):
            #The total size of trainPeople is inversely proportional to testPeople
            randomPosition = int(random.random()*(M))
            temporaryPerson = trainPeople[randomPosition]

            # Can not use person with only one image or less
            if len(temporaryPerson.getImages()) <= 1:
                continue

            randomImagePerson = int(random.random()*len(temporaryPerson.getImages()))

            testPeople.append(EigenPerson(name=temporaryPerson.getName(),
                            images=[temporaryPerson.getImages()[randomImagePerson]],
                            directory=temporaryPerson.getDirectory()))

            del(temporaryPerson.getImages()[randomImagePerson])

            # #The total size of trainFaces is inversely proportional to facesToTest
            # randomPosition = int(random.random()*(M-len(facesToTest)))
            #
            # facesToTest[i][:] = trainFaces[randomPosition][:]
            #
            # #0 is for the axis 0 (row)
            # trainFaces = np.delete(trainFaces, randomPosition, 0)
            #
            # #One face was removed from the matrix, so his length decrease too
            # numberOfFaces -= 1
        print 'Number of trainFaces and testFaces:', self.getNumberOfFaces(trainPeople), self.getNumberOfFaces(testPeople)
        return trainPeople, testPeople

#-------------------------------------------------------------------------------

    # Find a person by name in a set of people
    def __findPersonByName(self, people, name):
        # The names of people must be unique
        for person in people:
            if person.getName() == name:
                return person
        return 'Person not found in database'

#-------------------------------------------------------------------------------

    # With the map it's possible recover the face on the matrix of faces and
    # allocate to respective person
    def __getPersonByRowMatrix(self, people, row):
        for personMap in self.__peopleMap:
            for face in self.__peopleMap[personMap]:
                if face == row:
                    return self.__findPersonByName(people, personMap)

#-------------------------------------------------------------------------------
    # The eigenFace method must use matrix of faces, but I want to recover these
    # faces to get informations of the person found, so it's necessary map each
    # each person to your respective faces
    def __mapMatrixToPerson(self, initialIndex, finalIndex):
        faceVector = []
        for i in range(initialIndex, finalIndex):
            faceVector.append(i)

        return faceVector

#-------------------------------------------------------------------------------

    # Transforming the faces of each person on the matrix of faces.
    # In the case of training people, it is necessary to map them
    def __makeTrainFacesMatrix(self, people):
        faces = people[0].getFacesMatrix()

        self.__peopleMap[people[0].getName()] = self.__mapMatrixToPerson(0, len(faces)-1)

        for i in range(1, len(people)):

            initialIndex = len(faces) - 1
            faces = np.concatenate((faces, people[i].getFacesMatrix()), axis=0)
            finalIndex = len(faces) - 1

            self.__peopleMap[people[i].getName()] = self.__mapMatrixToPerson(initialIndex, finalIndex)

        return faces

#-------------------------------------------------------------------------------

    # Transforming the faces of each person on the matrix of faces.
    def __makeTestFacesMatrix(self, people):
        faces = people[0].getFacesMatrix()
        for i in range(1, len(people)):
            faces = np.concatenate((faces, people[i].getFacesMatrix()), axis=0)
        return faces

#-------------------------------------------------------------------------------

    #Make a vector with the mean of all columns
    def __averageVector(self, faces):
        (M, N) = np.shape(faces)
        average = np.zeros((N), dtype=np.float32)

        for j in range(N):
            for i in range(M):
                average[j] += faces[i][j]
            average[j] = average[j] / M

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

    #The main method
    def eigenFaceMethod(self, quantityPeopleToTest=1, precision=100, showResults=False):

        trainPeople, testPeople = self.__getRandomPeopleToTest(self.people, quantityPeopleToTest)
        trainPeople = self.__preparePeople(trainPeople)
        testPeople = self.__preparePeople(testPeople)

        trainFaces = self.__makeTrainFacesMatrix(trainPeople)
        testFaces = self.__makeTestFacesMatrix(testPeople)

        if self.__eigenFaces == None:

            averageVector = self.__averageVector(trainFaces)

            meanFaces = self.__removeMean(trainFaces, averageVector)

            self.__eigenFaces = self.getEigenFaces(meanFaces, precision)

        eigenTrainFaces = np.dot(self.__eigenFaces, meanFaces.transpose()) # 17x20
        eigenTrainFaces = eigenTrainFaces.transpose()

        meanTestFace = self.__removeMean(testFaces, averageVector)

        eigenTestFaces = np.dot(self.__eigenFaces, meanTestFace.transpose())
        eigenTestFaces = eigenTestFaces.transpose()

        euclideanDistances = self.__applyEuclidianDistance(eigenTrainFaces, eigenTestFaces)
        foundPeople = []

        for i in range(len(euclideanDistances)):
            posMinValue = np.argmin(euclideanDistances[i][:])

            #Comparing the testImage X foundPerson
            foundPerson = trainFaces[posMinValue][:]
            originalFace = testFaces[i][:]
            foundPerson = self.__getPersonByRowMatrix(trainPeople, posMinValue)
            if showResults == True:
                print 'The found person was: ', foundPerson.getName()
                compareImages((self.getImagePerson(foundPerson), self.getImagePerson(testPeople[i])))

            foundPeople.append(foundPerson)

        return foundPeople, testFaces

    #-------------------------------------------------------------------------------
