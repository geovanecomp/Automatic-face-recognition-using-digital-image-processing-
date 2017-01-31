# -*- coding: UTF-8 -*-
import time
import cv2
from Person import *
from Utils import *
from os import *
import numpy as np
# Constants
# URLTRAIN = 'Source/FEI/TrainDatabase/'
URLTRAIN = 'Source/CompactFEI_160x120/TrainDatabase/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/Laplacian/Grayscale/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/Laplacian/SuavizationFilter_3x3/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/HistogramEqualization/EqualizedDatabase/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/AllMethods/Grayscale/'
# URLTRAIN = 'Source/Bernardo/TrainDatabase/'


# URLTRAIN = 'Source/Bernardo/TrainDatabase/'

if __name__ == '__main__':

    from Recognition.Correlation import *
    from Recognition.EigenFace import *
    from ChainOfResponsibility.ImageProcessingChainOfResponsibility import *
    from ChainOfResponsibility.HistogramEqualization import *
    from ChainOfResponsibility.SuavizationFilter import *
    from ChainOfResponsibility.LaplacianFilter import *

    # print 'Escolha um método para o reconheicmento'
    # choice = raw_input()

    print 'Trainnig URL:', URLTRAIN
    initialTime = time.time()
# ------------------------------------------------------------------------------

    # CORRELATION METHOD
    # correlation = Correlation(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    # correlation.setFaceIndicesToTest([3, 7]) # To fixed people
    # correlation.setFacesPerPersonToTest(2) # To random people
    # foundPerson, testPeople, percentage = correlation.bruteForce(quantityPeopleToTest=3, showResults=True)

# #-------------------------------------------------------------------------------

    # EIGENFACES METHOD
    eigenFace = EigenFace(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    eigenFace.setFaceIndicesToTest([3, 7, 11, 13])
    # eigenFace.setFaceIndicesToTest([0, 4, 8, 12])
    foundPeople, testFaces = eigenFace.eigenFaceMethod(quantityPeopleToTest=5, precision=20, showResults=True)


#-------------------------------------------------------------------------------

    # CHAIN OF RESPONSIBILITY
    # All methods
    # chainOfResponsibility = ImageProcessingChainOfResponsibility(URLTRAIN,
    #     SuavizationFilter(
    #         LaplacianFilter(
    #             HistogramEqualization()
    #         ), dimensionMask=3
    #     ),
    #     channels=0)

    # Two methods
    # chainOfResponsibility = ImageProcessingChainOfResponsibility(URLTRAIN, SuavizationFilter(
    #         LaplacianFilter(), dimensionMask=3
    #     )
    # )

    # One method
    # chainOfResponsibility = ImageProcessingChainOfResponsibility(URLTRAIN, LaplacianFilter(), channels=0)
    #
    # chainOfResponsibility.setDirectory('Source/CompactFEI_320x240/Comparacoes/Laplaciano_com_mascara/')
    # chainOfResponsibility.setPeople(chainOfResponsibility.loadPeople(3))
    # chainOfResponsibility.calculate()

#-------------------------------------------------------------------------------

    print 'Past time:', time.time() - initialTime

#-------------------------------------------------------------------------------
