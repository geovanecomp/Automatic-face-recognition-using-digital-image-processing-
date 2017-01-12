# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class EigenPerson(Person):
    'This class represents one person with the requirements to eigenface method'

    def __init__(self, directory, name):
        super(EigenPerson, self).__init__(directory, name)
        self.__images = images


    def setImages(self, images):
        self.__images = images

    def getImages(self):
        return self.__images

    def addImage(self, image):
        self.__images.append(image)

    
