# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Person import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '_'
AVERAGE   = 'average'

class CompleteBruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, urlTestImage):
        self.testImage = cv2.imread(urlTestImage)
        self.testImage = np.float32(self.testImage)
        avg = self.__averageImage(self.testImage)
        self.personTest = Person(directory=urlTestImage, name='unknown', images=self.testImage, average=avg)
        self.M, self.N, self.O = self.testImage.shape

#-------------------------------------------------------------------------------

    #Get all people to compare
    def __getPeople(self):
        #Count the number of "people".
        #-1 its because this function count the TrainDatabase too
        numberOfFolders = len(list(os.walk(URLTRAIN))) - 1;
        people = [None] * numberOfFolders

        for i in range(numberOfFolders):

            #Getting the url, folders and files
            directory, folders, files = os.walk(URLTRAIN+str(i+1)).next()

            images = [None] * len(files)

            for (j, file) in enumerate(files):
                name, image = file.split(DELIMITER)
                images[j] = image

            person = Person(directory=directory, name=name, images=images)
            people[i] = person

        return people

#-------------------------------------------------------------------------------

    #Every person has an average of all his images
    def __averagePersonImage(self, people):

        for person in people:
            images = person.getImages()
            average = 0.0
            for imageName in images:
                imageUrl = person.getName()+DELIMITER+imageName

                image = cv2.imread(person.getDirectory()+'/'+imageUrl)

                average += self.__averageImage(image)

            average = average / float(len(images))
            person.setAverage(average)

        return people


    #Calculate the average of one image
    def __averageImage(self, image):
        #Instead I could use np.average(image)
        M, N, O = image.shape

        sumOfElements = 0.0
        numberElements = M*N*O

        for i in range(M):
            for j in range(N):
                for k in range(O):
                    sumOfElements += image[i][j][k]

        average = sumOfElements / numberElements

        return average

#-------------------------------------------------------------------------------
