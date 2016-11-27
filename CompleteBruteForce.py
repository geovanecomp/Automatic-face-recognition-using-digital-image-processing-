# -*- coding: UTF-8 -*-
import cv2
import numpy as np
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

class CompleteBruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, urlTestImage, channels=0):
        self.__people = None
        self.__channels = channels
        self.__testImage = readImage(urlTestImage, self.__channels)
        self.__testImage = np.float32(self.__testImage)
        print self.__testImage.shape
        try:
            self.M, self.N, self.O = self.__testImage.shape
        except:
            self.M, self.N = self.__testImage.shape
            self.O = 1
            self.__testImage = self.__testImage.reshape((self.M,self.N,self.O))

        avg = self.__averageImage(self.__testImage)
        self.personTest = Person(directory=urlTestImage, name='unknown', images=self.__testImage, average=avg)


    def setPeople(self, people):
        self.__people = people

    def getPeople(self):
        return self.__people
#-------------------------------------------------------------------------------

    #Every person has an average of all his images
    def __averagePersonImage(self, people):

        for person in people:
            images = person.getImages()
            average = 0.0
            for imageName in images:
                imageUrl = person.getName()+DELIMITER+imageName

                image = readImage(person.getDirectory()+'/'+imageUrl, self.__channels)
                average += self.__averageImage(image)

            average = average / float(len(images))
            person.setAverage(average)

        return people


    #Calculate the average of one image
    def __averageImage(self, image):
        #Instead I could use np.average(image)
        #self.M, self.N, self.O = image.shape

        sumOfElements = 0.0
        numberElements = self.M*self.N*self.O

        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    sumOfElements += image[i][j][k]

        average = sumOfElements / numberElements

        return average

#-------------------------------------------------------------------------------

    #This method create an average matrix of a person to optimize the correlation
    def __averageMatrix(self, person):

        #TODO: If an old average image exists, must be deleted
        # try:
        #     os.remove(os.path.join(person.getDirectory(), person.getName()+DELIMITER+AVERAGE+EXTENSION))
        # except:
        #     pass

        avgMatrix = np.zeros((self.M, self.N, self.O), dtype=np.float32)

        for imageName in person.getImages():
            imageUrl = person.getName()+DELIMITER+imageName
            image = readImage(person.getDirectory()+'/'+imageUrl, self.__channels)

            for i in range(self.M):
                for j in range(self.N):
                    for k in range(self.O):
                        avgMatrix[i][j][k] += image[i][j][k]

            avgMatrix = avgMatrix / len(person.getImages())

            #If I want to save the avgMatrix as image
            #cv2.imwrite(os.path.join(person.getDirectory(), person.getName()+DELIMITER+AVERAGE+EXTENSION), avgMatrix)

        return avgMatrix

#-------------------------------------------------------------------------------

    #This method will be responsible to make the comparison between two people
    def __correlation(self, testPerson, trainPerson):
        A = testPerson
        averageMatrixTrainPerson = self.__averageMatrix(trainPerson)

        avg1 = testPerson.getAverage()
        avg2 = trainPerson.getAverage()

        numerator    = 0
        denominator1 = 0
        denominator2 = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = testPerson.getImages()[i][j][k]
                    B = averageMatrixTrainPerson[i][j][k]

                    numerator    += (A - avg1) * (B - avg2)
                    denominator1 += (A - avg1)**2
                    denominator2 += (B - avg2)**2


        correlation = numerator / (denominator1 * denominator2)**0.5

        print "TrainPerson: ", trainPerson.getName() ," The images are " , correlation * 100, "% equals"
        return correlation

#-------------------------------------------------------------------------------

    #The main method
    def bruteForce(self):
        people = self.__averagePersonImage(self.__people)

        foundPerson = None
        maxCorrelation    = 0

        results = np.zeros(len(people))
        for (i, person) in enumerate(people):
            results[i] = self.__correlation(self.personTest, person)
            if results[i] > maxCorrelation:
                foundPerson     = person
                maxCorrelation  = results[i]

        return foundPerson, maxCorrelation
