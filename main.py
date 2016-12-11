# -*- coding: UTF-8 -*-
import time
import cv2
from Person import *
from Utils import *

URLTEST     = 'Source/Bernardo/TestDatabase/'
URLTRAIN    = 'Source/Bernardo/TrainDatabase/'
URLOTHERS   = 'Source/OthersImages/'
EXTENSION   = '.jpg'

def grayScale(image):
    try:
        M, N, O = image.shape

        grayImage = np.zeros((M, N, 1), dtype=np.uint8)
        initialTime = time.time()
        for i in range(M):
            for j in range(N):
                grayImage[i][j][0] = image[i][j][0] * 0.2989 + image[i][j][1] * 0.5870 + image[i][j][2] * 0.1140

        return grayImage

    except:
        return image


#Get all people to compare
def getPeople():
    #Count the number of "people".
    #-1 its because this function count the TrainDatabase too
    numberOfFolders = len(list(os.walk(URLTRAIN))) - 1;
    people = [None] * numberOfFolders

    for i in range(numberOfFolders):

        #Getting the url, folders and files
        directory, folders, files = os.walk(URLTRAIN+str(i+1)).next()

        images = [None] * len(files)

        for (j, file) in enumerate(files):
            name, image = file.split(DELIMITER)
            images[j] = image

        person = Person(directory=directory, name=name, images=images)
        people[i] = person

    return people


if __name__ == '__main__':
    from HistogramEqualization import *
    from LaplacianFilter import *
    from FourierTransform import *
    from FourierTransform2 import *
    from BruteForce import *
    from CompleteBruteForce import *
    from EigenFace import *


    initialTime = time.time()
    people = getPeople()

    # image1 = cv2.imread(URLOTHERS+'game1.png')
    # image2 = cv2.imread(URLOTHERS+'game2.png')
    #image = grayScale(image)

    #Here is another way to convert to grayscale using opencv
    #gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # histogramEqualization = HistogramEqualization(image)
    # equalizedImage = histogramEqualization.calculate(True)

    # laplacian = LaplacianFilter(image)
    # laplacian.laplacianFilter(True)

    # fourier = FourierTransform(image)
    # fourier.fourierTransform(1, 50, True)

    # completeBrute = CompleteBruteForce(urlTestImage=URLTEST+'10'+EXTENSION)
    # completeBrute.setPeople(getPeople())
    # foundPerson, percentage = completeBrute.bruteForce()
    # print 'The person found was:', foundPerson.getName(), 'with ', percentage*100, '% of accuracy'

    eigenFace = EigenFace(urlTestImage=URLTEST+'10'+EXTENSION)
    eigenFace.setPeople(people)
    eigenFace.eigenFaceMethod(20)

    print 'Past time:', time.time() - initialTime
