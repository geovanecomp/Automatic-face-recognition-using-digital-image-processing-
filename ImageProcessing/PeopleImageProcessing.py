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

class PeopleImageProcessing(object):
    "This class will responsible to apply image processing in people's face"

    def __init__(self, quantityPeople=None, channels=0):
        self.__peopleMap = {}
        self.__channels = channels
        self.people = self.loadPeople()
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

    # def applyHistogramEqualization(people):
    #     transformedPeople = [None] * len(people)
    #     for person in people:
    #         person.getI
