# -*- coding: UTF-8 -*-
import numpy as np
import os

from Recognizer import *
from Person.CorrelationPerson import *
from Utils import *

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#Constants
EXTENSION = '.jpg'
DELIMITER = '-'

class Correlation(Recognizer):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self,  urlTrain, quantityPeopleToTrain=None, channels=0):
        super(Correlation, self).__init__(urlTrain)
        self.channels = channels
        self.people = self.getPeople(quantityPeopleToTrain)
        self.M, self.N, self.O = self.people[0].getDimensionOfImage()


#-------------------------------------------------------------------------------

    #Get all people to compare
    def getPeople(self, numberOfPeople=None):

        if numberOfPeople == None:
            #-1 its because this function count the TrainDatabase too
            numberOfPeople = len(list(os.walk(self.urlTrain))) - 1;

        people = [None] * numberOfPeople

        for i in range(numberOfPeople):

            #Getting the url, folders and files
            directory, folders, files = os.walk(self.urlTrain+str(i+1)).next()

            images = [None] * len(files)

            for (j, file) in enumerate(files):
                name, image = file.split(DELIMITER)
                images[j] = image

            person = CorrelationPerson(name=name, images=images,
                    directory=directory, channels=self.channels)
            people[i] = person

        return people

#-------------------------------------------------------------------------------

    def __getFixedPeopleToTest(self, trainPeople, numberOfPeople=1):
        testPeople = []

        for i in range(numberOfPeople):
            for index in self.faceIndices:
                testPeople.append(CorrelationPerson(name=trainPeople[i].getName(),
                                images=[trainPeople[i].getImages()[index]],
                                directory=trainPeople[i].getDirectory(),
                                channels=self.channels))



        # After allocating the people to test, I need remove them from trainPeople
        # Starting from the last element to do not re-allocate the array
        for i in range(numberOfPeople):
            for j in reversed(range(len(self.faceIndices))):
                del(trainPeople[i].getImages()[self.faceIndices[j]])

        return trainPeople, testPeople


#-------------------------------------------------------------------------------

    # Select (and remove) randomly a number of faces from train database to test
    # It's important to know that need to remove an image from a person and NOT,
    # the person from training database
    def __getRandomPeopleToTest(self, trainPeople, numberOfPeople=1):

        M = len(trainPeople)

        testPeople = []

        temporaryPerson = None
        for i in range(numberOfPeople):

            #The total size of trainPeople is inversely proportional to testPeople
            randomPosition = int(random.random()*(M-len(testPeople)))

            temporaryPerson = trainPeople[randomPosition]

            # Can not use person with only one image or less
            if len(temporaryPerson.getImages()) <= 1:
                continue

            # Getting random faces too
            for j in range(self.numberFacesToTest):
                randomImagePerson = int(random.random()*len(temporaryPerson.getImages()))

                testPeople.append(CorrelationPerson(name=temporaryPerson.getName(),
                                images=[temporaryPerson.getImages()[randomImagePerson]],
                                directory=temporaryPerson.getDirectory(), channels=self.channels))

                del(temporaryPerson.getImages()[randomImagePerson])

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

        # print "TrainPerson: ", trainPerson.getName() ,"  The images are " , correlation * 100, "% equals"
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
        numberOfErros = 0.0
        for i in range(len(maxCorrelations)):
            if foundPerson[i].getName() != testPeople[i].getName():
                print 'Name of the found person: ', foundPerson[i].getName()
                print 'Name of the test person: ', testPeople[i].getName()
                numberOfErros = numberOfErros + 1
            if maxCorrelations[i] < threshold:
                print "The test person number ", i, " was not found (",maxCorrelations[i]*100,"% of accuracy)"

            else:
                # testPeople[i].setName(foundPerson[i].getName())
                print "The person found was: ", foundPerson[i].getName(), "with ", maxCorrelations[i]*100, '% of accuracy'

            compareImages((foundPerson[i].loadFirstImage(), testPeople[i].loadFirstImage()))

        print 'The algorithm found correctly ', 100*(len(testPeople) - numberOfErros)/len(testPeople), '% of people'
        return testPeople

#-------------------------------------------------------------------------------

    #The main method
    def bruteForce(self, quantityPeopleToTest=1, threshold=60, showResults=True):

        if self.faceIndices != None:
            trainPeople, testPeople = self.__getFixedPeopleToTest(self.people, quantityPeopleToTest)
        else:
            trainPeople, testPeople = self.__getRandomPeopleToTest(self.people, quantityPeopleToTest)

        print 'Number of trainFaces and testFaces:', self.getNumberOfFaces(trainPeople), self.getNumberOfFaces(testPeople)

        people = self.__averagePersonImage(self.people)

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


        if showResults == True:
            testPeople = self.analyseResults(foundPerson, testPeople, maxCorrelations, threshold)

        return foundPerson, testPeople, maxCorrelations
