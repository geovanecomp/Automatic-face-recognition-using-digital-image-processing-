# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Person import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')
URLTRAIN  = 'Source/TrainDatabase/'
EXTENSION = '.jpg'

class CompleteBruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, testImage):
        self.testImage = np.float32(testImage)
        self.M, self.N, self.O = self.testImage.shape

    def __getPeople(self):
        #Count the number of "people".
        #-1 its because this function count the TrainDatabase too
        numberOfFolders = len(list(os.walk('Source/TrainDatabase'))) - 1;
        people = [None] * numberOfFolders



    def __averagePersonImage(self, image):




    def __averageImage(self, image):
        #Instead I could use np.average(image)
        M, N, O = image.shape

        sumOfElements = 0
        numberElements = M*N*O

        for i in range(M):
            for j in range(N):
                for k in range(O):
                    sumOfElements += image[i][j][k]

        average = sumOfElements / numberElements

        return average

    def bruteForce(self):
        avg1 = self.__averageImage(self.testImage)
        avg2 = self.__averageImage(self.image2)
        A = self.testImage
        B = self.image2
        print self.testImage.shape, self.image2.shape

#        correlationMatrix =np.zeros((self.M, self.N, self.O), dtype=np.float32)
        # raise Exception("TEXTO")
        numerator    = 0
        denominator1 = 0
        denominator2 = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = self.testImage[i][j][k]
                    B = self.image2[i][j][k]

                    numerator    += (A - avg1) * (B - avg2)
                    denominator1 += (A - avg1)**2
                    denominator2 += (B - avg2)**2


        correlation = numerator / (denominator1 * denominator2)**0.5

        print "The images are " , correlation * 100, "% equals"
        return correlation


















    #asdas
