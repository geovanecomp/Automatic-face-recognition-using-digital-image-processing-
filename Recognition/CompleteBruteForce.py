# -*- coding: UTF-8 -*-
import numpy as np
import os

from Recognizer import *
from Person.CorrelationPerson import *
from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
URLTRAIN  = 'Source/Bernardo/TrainDatabase/'
# URLTRAIN    = 'Source/CompactFEI_160x120/TrainDatabase/'
# URLTRAIN    = 'Source/CompactFEI_80x60/TrainDatabase/'
EXTENSION = '.jpg'
DELIMITER = '-'

class CompleteBruteForce(Recognizer):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, channels=0):
        super(CompleteBruteForce, self).__init__()
        self.__channels = channels
        self.people = self.getPeople()
        self.M, self.N, self.O = self.people[0].getDimensionOfImage()

#-------------------------------------------------------------------------------

    #Get all people to compare
    def getPeople(self, numberOfPeople=None):
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

            person = CorrelationPerson(name=name, images=images,
                    directory=directory, channels=self.__channels)
            people[i] = person

        return people

#-------------------------------------------------------------------------------

    def __getFixedPeopleToTest(self, trainPeople, numberOfPeople=1):
        testPeople = [None] * numberOfPeople

        # Default index to get all fixed images
        imgIndex = 0

        for i in range(numberOfPeople):
            testPeople[i] = CorrelationPerson(name=trainPeople[i].getName(),
                            images=[trainPeople[i].getImages()[imgIndex]],
                            directory=trainPeople[i].getDirectory(),
                            channels=self.__channels)

            del(trainPeople[i].getImages()[imgIndex])

        return trainPeople, testPeople


#-------------------------------------------------------------------------------

    # Select (and remove) randomly a number of faces from train database to test
    # It's important to know that need to remove an image from a person and NOT,
    # the person from training database
    def __getRandomPeopleToTest(self, trainPeople, numberOfPeople=1):

        M = len(trainPeople)

        # Setting some fixed faces to analyse results (Temporary - To TCC)
        # trainPeople, testPeople = self.__getFixedPeopleToTest(trainPeople, numberOfPeople)

        testPeople = [None] * numberOfPeople

        temporaryPerson = None
        for i in range(numberOfPeople):

            #The total size of trainPeople is inversely proportional to testPeople
            randomPosition = int(random.random()*(M-len(testPeople)))

            temporaryPerson = trainPeople[randomPosition]

            # Can not use person with only one image or less
            if len(temporaryPerson.getImages()) <= 1:
                continue

            randomImagePerson = int(random.random()*len(temporaryPerson.getImages()))

            testPeople[i] = CorrelationPerson(name=temporaryPerson.getName(),
                            images=[temporaryPerson.getImages()[randomImagePerson]],
                            directory=temporaryPerson.getDirectory(), channels=self.__channels)

            del(temporaryPerson.getImages()[randomImagePerson])


        print 'Number of trainFaces and testFaces:', self.getNumberOfFaces(trainPeople), self.getNumberOfFaces(testPeople)
        return trainPeople, testPeople


#-------------------------------------------------------------------------------

    #Every person has an average of all his images
    def __averagePersonImage(self, people):

        for person in people:
            average = 0.0
            images = person.loadImages()

            for image in images:
                average += self.__averageImage(image)

            average = average / float(len(images))
            person.setAverage(average)

        return people


    #Calculate the average of one image
    def __averageImage(self, image):
        #Instead I could use np.average(image)
        #self.M, self.N, self.O = image.shape

        sumOfElements = 0.0
        numberElements = self.M*self.N*self.O

        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    sumOfElements += image[i][j][k]

        average = sumOfElements / numberElements

        return average

#-------------------------------------------------------------------------------

    #This method create an average matrix of a person to optimize the correlation
    def __averageMatrix(self, person):

        #TODO: If an old average image exists, that must be deleted
        # try:
        #     os.remove(os.path.join(person.getDirectory(), person.getName()+DELIMITER+AVERAGE+EXTENSION))
        # except:
        #     pass

        avgMatrix = np.zeros((self.M, self.N, self.O), dtype=np.float32)

        images = person.loadImages()
        for image in images:

            for i in range(self.M):
                for j in range(self.N):
                    for k in range(self.O):
                        avgMatrix[i][j][k] += image[i][j][k]

            avgMatrix = avgMatrix / len(person.getImages())

            #If I want to save the avgMatrix as image
            #cv2.imwrite(os.path.join(person.getDirectory(), person.getName()+DELIMITER+AVERAGE+EXTENSION), avgMatrix)

        return avgMatrix

#-------------------------------------------------------------------------------

    #This method will be responsible to make the comparison between two people
    def __correlation(self, trainPerson, testPerson):
        averageMatrixTrainPerson = self.__averageMatrix(trainPerson)
        averageMatrixTestPerson = self.__averageMatrix(testPerson)

        avg1 = testPerson.getAverage()
        avg2 = trainPerson.getAverage()

        numerator    = 0
        denominator1 = 0
        denominator2 = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = averageMatrixTestPerson[i][j][k]
                    B = averageMatrixTrainPerson[i][j][k]

                    numerator    += (A - avg1) * (B - avg2)
                    denominator1 += (A - avg1)**2
                    denominator2 += (B - avg2)**2


        correlation = numerator / (denominator1 * denominator2)**0.5

        print "TrainPerson: ", trainPerson.getName() ," The images are " , correlation * 100, "% equals"
        return correlation

#-------------------------------------------------------------------------------
    def analyseResults(self, foundPerson, testPeople, maxCorrelations, threshold):
        # Every person have the triple information:
        # herself (testPeople[i]), foundPerson[i] with max correlation
        # and the correlation[i] between herself and the foundPerson. The triple
        # must have the same length.
        if len(foundPerson) != len(testPeople) and len(testPeople) != len(maxCorrelations):
            raise "The number of the vectors to compare are different"

        # If threshold is in %, there must be converted to decimal value
        if threshold > 1.0 :
            threshold = threshold / 100.0

        # If the correlation is lower than a specified value, the found person
        # is wrong.
        for i in range(len(maxCorrelations)):
            if maxCorrelations[i] < threshold:
                print "The test person number ", i, " was not found (",maxCorrelations[i]*100,"% of accuracy)"

            else:
                testPeople[i].setName(foundPerson[i].getName())
                print "The person found was: ", testPeople[i].getName(), "with ", maxCorrelations[i]*100, '% of accuracy'

            compareImages((foundPerson[i].loadFirstImage(), testPeople[i].loadFirstImage()))

        return testPeople

#-------------------------------------------------------------------------------

    #The main method
    def bruteForce(self, quantityPeopleToTest=1, threshold=60):
        self.people, testPeople = self.__getRandomPeopleToTest(self.people, quantityPeopleToTest)

        people = self.__averagePersonImage(self.people)
        print 'vai passar a teste people:'
        testPeople = self.__averagePersonImage(testPeople)

        foundPerson = [None] * len(testPeople)
        maxCorrelations = np.zeros(len(testPeople))

        results = np.zeros((len(testPeople), len(people)))
        for (i, testPerson) in enumerate(testPeople):
            for (j, person) in enumerate(people):
                results[i][j] = self.__correlation(person, testPerson)
                if results[i][j] > maxCorrelations[i]:
                    foundPerson[i]     = person
                    maxCorrelations[i]  = results[i][j]

        testPeople = self.analyseResults(foundPerson, testPeople, maxCorrelations, threshold)

        return foundPerson, testPeople, maxCorrelations
