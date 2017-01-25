# -*- coding: UTF-8 -*-

import random
from abc import ABCMeta, abstractmethod
from Utils import *

#Constants
DELIMITER = '-'

class Recognizer(object):
    __metaclass__ = ABCMeta
    'This class will provide the basis for all recognitions'

    def __init__(self, urlTrain):
        self.people = None
        self.urlTrain = urlTrain

        # Used to set the number of faces in the random method
        self.numberFacesToTest = 1

        # If it's necessary fix some faces to test
        self.faceIndices = None

#-------------------------------------------------------------------------------

    #Get the people from the database
    @abstractmethod
    def getPeople(self, numberOfPeople=None):
        pass

#-------------------------------------------------------------------------------

    def setPeople(self, people):
        self.people = people

#-------------------------------------------------------------------------------

    def setUrlTrain(self, urlTrain):
        self.urlTrain = urlTrain

#-------------------------------------------------------------------------------

    def getNumberOfFaces(self, people):
        count = 0
        for person in people:
            count = len(person.getImages()) + count

        return count

#-------------------------------------------------------------------------------

    def setFacesPerPersonToTest(self, numberFacesToTest):
        self.numberFacesToTest = numberFacesToTest

#-------------------------------------------------------------------------------

    def setFaceIndicesToTest(self, faceIndices):
        self.faceIndices = faceIndices

#-------------------------------------------------------------------------------
