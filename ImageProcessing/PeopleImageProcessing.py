# -*- coding: UTF-8 -*-
import numpy as np
import os

from ImageProcessing.HistogramEqualization import *
from ImageProcessing.LaplacianFilter import *
from Person.Person import *
from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
# URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
URLTRAIN    = 'Source/CompactFEI_80x60/TrainDatabase/'
# URLTRAIN    = 'Source/CompactFEI_320x240/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '-'
AVERAGE   = 'average'

class PeopleImageProcessing(object):
    "This class will responsible to apply image processing in people's face"

    def __init__(self, directory, quantityPeople=None, channels=3):
        self.__directory = directory
        self.__channels = channels
        self.people = self.loadPeople(quantityPeople)
        self.M, self.N, self.O = self.people[0].getDimensionOfImage()

    def setPeople(self, people):
        self.people = people

    def getPeople(self):
        return self.people

#-------------------------------------------------------------------------------

    def loadPeople(self, numberOfPeople=None):
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

            person = Person(name=name, images=images, directory=directory,
                    channels=self.__channels)
            people[i] = person

        return people
#-------------------------------------------------------------------------------

    def applyHistogramEqualization(self, people):
        transformedPeople = [None] * len(people)
        histogramEqualization = HistogramEqualization()
        print 'Number of people to apply Histogram Equalization: ', len(people)
        for k, person in enumerate(people):
            print "Person nº: ",k
            imageNames = person.getImages()
            images = person.loadImages()
            partialDirectory = self.__directory + person.getName() + '/' + person.getName()

            for i, image in enumerate(images):
                directory = partialDirectory + DELIMITER + imageNames[i]
                equalizedImage = histogramEqualization.calculate(image)
                # equalizedImage = cv2.equalizeHist(image)
                cv2.imwrite(directory, equalizedImage)

    def applyLaplacianFilter(self, people):
        transformedPeople = [None] * len(people)
        laplacian = LaplacianFilter()
        print 'Number of people to apply Laplacian Filter: ', len(people)
        for k, person in enumerate(people):
            print "Person nº: ",k
            imageNames = person.getImages()
            images = person.loadImages()
            partialDirectory = self.__directory + person.getName() + '/' + person.getName()

            for i, image in enumerate(images):
                directory = partialDirectory + DELIMITER + imageNames[i]
                laplacianImage = laplacian.laplacianFilter(image)[1]
                cv2.imwrite(directory, laplacianImage)
