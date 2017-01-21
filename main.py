# -*- coding: UTF-8 -*-
import time
import cv2
from Person import *
from Utils import *
from os import *

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

    # CORRELATION METHOD
    # completeBrute = CompleteBruteForce(channels=3)
    # foundPerson, testPeople, percentage = completeBrute.bruteForce(quantityPeopleToTest=2)

    # EIGENFACES METHOD
    # eigenFace = EigenFace(quantityPeopleToTrain=20, channels=3)
    # foundPeople = eigenFace.eigenFaceMethod(quantityPeopleToTest=5,precision=100, showResults=True)

    # IMAGE PROCESSING
    # directory = 'Source/CompactFEI_80x60/ImageProcessing/HistogramEqualization/EqualizedDatabaseColorful/'
    #
    # imageProcessing = PeopleImageProcessing(directory)
    # people = imageProcessing.getPeople()
    # imageProcessing.applyHistogramEqualization(people)
    # imageProcessing.applyLaplacianFilter(people)

    # CHAIN OF RESPONSIBILITY
    chainOfResponsibility = PeopleChainOfResponsibility(HistogramEqualization(
        SuavizationFilter(
            LaplacianFilter()
        )
    ), channels=3)

    # chainOfResponsibility = PeopleChainOfResponsibility(SuavizationFilter(
    #         LaplacianFilter()
    #     )
    # )
    chainOfResponsibility.setDirectory('Source/Bernardo/ImageProcessing/AllMethods/Colorful/')
    chainOfResponsibility.setPeople(chainOfResponsibility.loadPeople())
    chainOfResponsibility.calculate()

    print 'Past time:', time.time() - initialTime

#-------------------------------------------------------------------------------
