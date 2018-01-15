# -*- coding: UTF-8 -*-
import time
import cv2
from Person import *
from Utils import *
from os import *
import numpy as np
# Constants
# URLTRAIN = 'Source/FEI/TrainDatabase/'
URLTRAIN = 'Source/CompactFEI_80x60/TrainDatabase/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/Laplacian/Grayscale/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/Laplacian/SuavizationFilter_3x3/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/HistogramEqualization/EqualizedDatabase/'
# URLTRAIN = 'Source/CompactFEI_80x60/ImageProcessing/AllMethods/Grayscale/'
# URLTRAIN = 'Source/Bernardo/TrainDatabase/'


# URLTRAIN = 'Source/Bernardo/TrainDatabase/'

def standardDeviation (average, results):
    differenceSquare = []
    for i in range(results):
        differenceSquare.append( (average - results[i]) **2 )
    return sum(differenceSquare) ** 0.5

def runEigenfacesInLot(numberOfTests=100, quantityPeopleToTest=5, precision=50, showResults=False):
    totalAverage = 0.0
    partialResults = []

    for i in range(numberOfTests):
        print '--------------------------- Test Nº: ', i
        foundPeople, testFaces, successRate = runEigenFace(quantityPeopleToTest, precision, showResults)
        partialResults.append(successRate)
        totalAverage += successRate

    totalAverage = totalAverage / numberOfTests

    print 'The partial results: ', partialResults
    print 'The total average is: ', totalAverage
    print 'The standardDeviation (NP) is: ', np.std(partialResults)
    return totalAverage

def runEigenFace(quantityPeopleToTest=5, precision=50, showResults=False):
    eigenFace = EigenFace(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    eigenFace.setFacesPerPersonToTest(4) # To random people
    return eigenFace.eigenFaceMethod(quantityPeopleToTest, precision, showResults)

def eigenFaceMethod():
    print 'How many people do you want to test?'
    quantityPeopleToTest = int(raw_input())

    print 'Select the precision method (1~100):'
    precision = int(raw_input())

    print 'Do you want to show all results? (1 or 0)'
    showResults = bool(raw_input())

    print 'Do you want to run Eigenfaces in lot? (1 or 0)'
    shouldRunInLot = bool(raw_input())

    if shouldRunInLot:
        print 'How many tests do you want to do?'
        numberOfTests = int(raw_input())
        runEigenfacesInLot(numberOfTests, quantityPeopleToTest, precision, showResults)
    else:
        runEigenFace(quantityPeopleToTest, precision, showResults)

# CORRELATION METHOD
def correlationMethod():
    correlation = Correlation(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    correlation.setFaceIndicesToTest([3, 7]) # To fixed people
    correlation.setFacesPerPersonToTest(2) # To random people
    foundPerson, testPeople, percentage = correlation.bruteForce(quantityPeopleToTest=3, showResults=True)

if __name__ == '__main__':

    from Recognition.Correlation import *
    from Recognition.EigenFace import *
    from ChainOfResponsibility.ImageProcessingChainOfResponsibility import *
    from ChainOfResponsibility.HistogramEqualization import *
    from ChainOfResponsibility.SuavizationFilter import *
    from ChainOfResponsibility.LaplacianFilter import *

    print 'Choose a Method:'
    print 'Choose (1) for Eigenfaces and (2) for Correlation or (3) for Image Processing:'
    choice = int(raw_input())

    print 'Trainnig URL:', URLTRAIN

    initialTime = time.time()
    if choice == 1:
        eigenFaceMethod()
    elif choice == 2:
        correlationMethod()
    elif choice == 3:
        correlationMethod()

    # EIGENFACES METHOD
    # eigenFace = EigenFace(urlTrain=URLTRAIN, quantityPeopleToTrain=25, channels=0)
    # eigenFace.setFaceIndicesToTest([3, 7, 11, 13])
    # eigenFace.setFaceIndicesToTest([0, 4, 8, 12])
    # foundPeople, testFaces = eigenFace.eigenFaceMethod(quantityPeopleToTest=5, precision=20, showResults=True)

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
