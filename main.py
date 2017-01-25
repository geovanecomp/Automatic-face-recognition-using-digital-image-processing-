# -*- coding: UTF-8 -*-
import time
import cv2
from Person import *
from Utils import *
from os import *

# Constants
URLTRAIN = 'Source/CompactFEI_80x60/TrainDatabase/'
# URLTRAIN = 'Source/CompactFEI_320x240/TrainDatabase/'
# URLTRAIN = 'Source/CompactFEI_320x240/ImageProcessing/Laplacian/SuavizationFilter_3x3/'
# URLTRAIN = 'Source/CompactFEI_320x240/ImageProcessing/Laplacian/Grayscale/'
# URLTRAIN = 'Source/Bernardo/TrainDatabase/'

if __name__ == '__main__':

    from Recognition.CompleteBruteForce import *
    from Recognition.EigenFace import *
    from ImageProcessing.PeopleImageProcessing import *
    from ChainOfResponsibility.PeopleChainOfResponsibility import *
    from ChainOfResponsibility.HistogramEqualization import *
    from ChainOfResponsibility.SuavizationFilter import *
    from ChainOfResponsibility.LaplacianFilter import *

    # print 'Escolha um método para o reconheicmento'
    # choice = raw_input()

    initialTime = time.time()
#-------------------------------------------------------------------------------
    # CORRELATION METHOD
    print 'URL de treino:', URLTRAIN
    completeBrute = CompleteBruteForce(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    # completeBrute.setFaceIndicesToTest([3, 7, 11, 13])
    completeBrute.setFaceIndicesToTest([0, 4, 8, 12])
    foundPerson, testPeople, percentage = completeBrute.bruteForce(quantityPeopleToTest=5, showResults=False)

#-------------------------------------------------------------------------------

    # EIGENFACES METHOD
    # eigenFace = EigenFace(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    # eigenFace.setFaceIndicesToTest([3, 7, 11, 13])
    # foundPeople = eigenFace.eigenFaceMethod(quantityPeopleToTest=5, precision=50, showResults=True)

#-------------------------------------------------------------------------------

    # IMAGE PROCESSING
    # directory = 'Source/CompactFEI_80x60/ImageProcessing/HistogramEqualization/EqualizedDatabaseColorful/'
    #
    # imageProcessing = PeopleImageProcessing(URLTRAIN, directory)
    # people = imageProcessing.getPeople()
    # imageProcessing.applyHistogramEqualization(people)
    # imageProcessing.applyLaplacianFilter(people)

#-------------------------------------------------------------------------------

    # CHAIN OF RESPONSIBILITY
    # chainOfResponsibility = PeopleChainOfResponsibility(
    #     SuavizationFilter(
    #         LaplacianFilter(
    #             HistogramEqualization()
    #         ), dimensionMask=3
    #     ),
    #     channels=0)

    # chainOfResponsibility = PeopleChainOfResponsibility(SuavizationFilter(
    #         LaplacianFilter(), dimensionMask=3
    #     )
    # )
    #
    # chainOfResponsibility = PeopleChainOfResponsibility(HistogramEqualization(), channels=0)
    #
    # chainOfResponsibility.setDirectory('Source/CompactFEI_320x240/ImageProcessing/HistogramEqualization/EqualizedDatabase/')
    # chainOfResponsibility.setPeople(chainOfResponsibility.loadPeople())
    # chainOfResponsibility.calculate()

#-------------------------------------------------------------------------------

    print 'Past time:', time.time() - initialTime

#-------------------------------------------------------------------------------
