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
        super(CompleteBruteForce, self).__init__(channels)
        self.people = self.getPeople()
        self.M, self.N, self.O = self.setDimensionsOfImage(self.people)

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

            person = CorrelationPerson(directory=directory, name=name, images=images)
            people[i] = person

        return people

#-------------------------------------------------------------------------------

    # Select (and remove) randomly a number of faces from train database to test
    # It's important to know that need to remove an image from a person and NOT,
    # the person from training database
    def __getRandomPeopleToTest(self, trainPeople, numberOfPeople=1):

        testPeople = [None] * numberOfPeople

        temporaryPerson = None
        for i in range(numberOfPeople):

            #The total size of trainPeople is inversely proportional to testPeople
            randomPosition = int(random.random()*(M-len(testPeople)))

            temporaryPerson = trainPeople[randomPosition]
            randomImagePerson = int(random.random()*len(temporaryPerson.getImages()))

            testPeople[i] = CorrelationPerson(name=temporaryPerson.getName(),
                            images=[temporaryPerson.getImages()[randomImagePerson]],
                            directory=temporaryPerson.getDirectory())

            del(temporaryPerson.getImages()[randomImagePerson])


        print 'Dimensions of train and test', np.shape(trainPeople), np.shape(testPeople)
        return trainPeople, testPeople


#-------------------------------------------------------------------------------

    #Every person has an average of all his images
    def __averagePersonImage(self, people):

        for person in people:
            images = person.getImages()
            average = 0.0

            for imageName in images:
                imageUrl = person.getName()+DELIMITER+imageName

                image = readImage(person.getDirectory()+'/'+imageUrl, self.channels)
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

        for imageName in person.getImages():
            imageUrl = person.getName()+DELIMITER+imageName
            image = readImage(person.getDirectory()+'/'+imageUrl, self.channels)

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


    #The main method
    def bruteForce(self, numberOfPeopleToTest=3, threshold=60):
        self.people, testPeople = self.__getRandomPeopleToTest(self.people, numberOfPeopleToTest)

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

        return foundPerson, testPeople, maxCorrelations
