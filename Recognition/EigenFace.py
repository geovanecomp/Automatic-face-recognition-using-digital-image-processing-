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
DELIMITER = '-'

class EigenFace(Recognizer):
    'This class will extract the main components of a image using PCA '

    def __init__(self, urlTrain, quantityPeopleToTrain=None, channels=0):
        super(EigenFace, self).__init__(urlTrain, channels)
        #I'll use a dictionary to map people to your faces on matrix of faces
        self.__peopleMap = {}
        self.__eigenFaces = None

        self.people = self.getPeople(quantityPeopleToTrain)
        self.M, self.N, self.O = self.people[0].getDimensionOfImage()



    def setPeople(self, people):
        self.people = people

#-------------------------------------------------------------------------------

    def getPeople(self, numberOfPeople=None):
        if numberOfPeople == None:
            #-1 its because this function count the TrainDatabase too
            numberOfPeople = len(list(os.walk(self.urlTrain))) - 1;
        people = [None] * numberOfPeople

        for i in range(numberOfPeople):

            #Getting the url, folders and files
            directory, folders, files = os.walk(self.urlTrain+str(i+1)).next()

            images = [None] * len(files)

            for (j, file) in enumerate(files):
                name, image = file.split(DELIMITER)
                images[j] = image

            person = EigenPerson(name=name, images=images, directory=directory,
                    channels=self.channels)
            people[i] = person

        return people
#-------------------------------------------------------------------------------

    #Get the trainFaces from database and append into a list to apply eigenfaces
    # method under her faces
    def __preparePeople(self, people):
        for person in people:
            person.setFacesMatrix()

        return people

#-------------------------------------------------------------------------------

    def __getFixedPeopleToTest(self, trainPeople, numberOfPeople=1):
        testPeople = []

        for i in range(numberOfPeople):
            for index in self.faceIndices:
                testPeople.append(EigenPerson(name=trainPeople[i].getName(),
                                images=[trainPeople[i].getImages()[index]],
                                directory=trainPeople[i].getDirectory(),
                                channels=self.channels))


        # After allocating the people to test, I need remove them from trainPeople
        # Starting from the last element to do not re-allocate the array
        for i in range(numberOfPeople):
            for j in reversed(range(len(self.faceIndices))):
                del(trainPeople[i].getImages()[self.faceIndices[j]])

        return trainPeople, testPeople

#-------------------------------------------------------------------------------

    #Select (and remove) randomly a number of faces from train database to test
    def __getRandomPeopleToTest(self, trainPeople, numberOfPeople=1):

        M = len(trainPeople)
        testPeople = []
        temporaryPerson = None

        for i in range(numberOfPeople):
            #The total size of trainPeople is inversely proportional to testPeople
            randomPosition = int(random.random()*(M))
            temporaryPerson = trainPeople[randomPosition]

            # Can not use person with only one image or less
            if len(temporaryPerson.getImages()) <= 1:
                continue

            # Getting random faces too
            for j in range(self.numberFacesToTest):

                randomImagePerson = int(random.random()*len(temporaryPerson.getImages()))

                testPeople.append(EigenPerson(name=temporaryPerson.getName(),
                                images=[temporaryPerson.getImages()[randomImagePerson]],
                                directory=temporaryPerson.getDirectory(),
                                channels=self.channels))

                del(temporaryPerson.getImages()[randomImagePerson])

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

        self.__peopleMap[people[0].getName()] = self.__mapMatrixToPerson(0, len(faces))

        for i in range(1, len(people)):

            initialIndex = len(faces)
            faces = np.concatenate((faces, people[i].getFacesMatrix()), axis=0)
            finalIndex = len(faces)

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
    def __showResults(self, trainFace, testFace):
        trainFace = trainFace.reshape(self.M, self.N, self.O)
        trainFace = correctMatrixValues(trainFace)

        testFace = testFace.reshape(self.M, self.N, self.O)
        testFace = correctMatrixValues(testFace)

        compareImages((trainFace,testFace))

#-------------------------------------------------------------------------------

    #The main method
    def eigenFaceMethod(self, quantityPeopleToTest=1, precision=100, showResults=False):

        # If an index was setted, it's necessary to use fixed faces
        if self.faceIndices != None:
            trainPeople, testPeople = self.__getFixedPeopleToTest(self.people, quantityPeopleToTest)
        else:
            trainPeople, testPeople = self.__getRandomPeopleToTest(self.people, quantityPeopleToTest)

        print 'Number of trainFaces and testFaces:', self.getNumberOfFaces(trainPeople), self.getNumberOfFaces(testPeople)

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

        numberOfErros = 0.0
        for i in range(len(euclideanDistances)):
            minPosError = np.argmin(euclideanDistances[i][:])

            #Comparing the foundPerson X testImage
            foundPerson = self.__getPersonByRowMatrix(trainPeople, minPosError)

            if showResults == True and foundPerson != None:

                print 'The found person was: ', foundPerson.getName(), 'The test was: ', testPeople[i].getName()
                print 'Min error: ', euclideanDistances[i][minPosError]
                print 'Max error:', euclideanDistances[i][np.argmax(euclideanDistances[i][:])]

                self.__showResults(trainFaces[minPosError][:], testFaces[i][:])
                # compareImages((foundPerson.loadFirstImage(), testPeople[i].loadFirstImage()))

                if foundPerson.getName() != testPeople[i].getName():
                    numberOfErros = numberOfErros + 1

            foundPeople.append(foundPerson)

        print 'The algorithm found correctly ', 100*(len(testPeople) - numberOfErros)/len(testPeople), '% of people'
        return foundPeople, testFaces

    #-------------------------------------------------------------------------------
