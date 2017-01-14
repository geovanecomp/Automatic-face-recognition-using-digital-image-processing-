# -*- coding: UTF-8 -*-

import random
from abc import ABCMeta, abstractmethod
from Utils import *

#Constants
DELIMITER = '-'

class Recognizer(object):
    __metaclass__ = ABCMeta
    'This class will provide the basis for all recognitions'

    def __init__(self, channels=0):
        self.people = None
        self.channels = channels

#-------------------------------------------------------------------------------

    #Get the people from the database
    @abstractmethod
    def getPeople(self, numberOfPeople=None):
        pass

#-------------------------------------------------------------------------------

    def setPeople(self, people):
        self.people = people

#-------------------------------------------------------------------------------

    # Get one person from people and set the default dimensions
    def setDimensionsOfImage(self, people):
        self.O = 1
        image = self.getImagePerson(people[0])
        self.M, self.N, self.O = image.shape

        return self.M, self.N, self.O

#-------------------------------------------------------------------------------

    def getImagePerson(self, person):

        imageName = person.getImages()[0]
        imageUrl = person.getName()+DELIMITER+imageName

        image = readImage(person.getDirectory()+'/'+imageUrl, self.channels)
        image = correctMatrixValues(image)

        return image

#-------------------------------------------------------------------------------

    def getNumberOfFaces(self, people):
        count = 0
        for person in people:
            count = len(person.getImages()) + count

        return count

#-------------------------------------------------------------------------------
