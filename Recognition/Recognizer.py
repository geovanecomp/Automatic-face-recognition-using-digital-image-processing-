# -*- coding: UTF-8 -*-

import random
from abc import ABCMeta, abstractmethod
from Utils import *

#Constants
DELIMITER = '-'

class Recognizer(object):
    __metaclass__ = ABCMeta
    'This class will provide the basis for all recognitions'

    def __init__(self):
        self.people = None

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
    # def setDimensionsOfImage(self, people):
    #     self.O = 1
    #     image = self.getImagePerson(people[0])
    #     self.M, self.N, self.O = image.shape
    #
    #     return self.M, self.N, self.O

#-------------------------------------------------------------------------------

    def getNumberOfFaces(self, people):
        count = 0
        for person in people:
            count = len(person.getImages()) + count

        return count

#-------------------------------------------------------------------------------
