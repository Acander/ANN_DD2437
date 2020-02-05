import numpy as np


def importAnimalDataSet():
    namesString = open("data_lab2/animalnames.txt", 'r').read()
    names = np.array(list(namesString.split("\n")))

    attributesString = open("data_lab2/animals.dat", 'r').read()
    attributeArray = np.array(list(map(int, attributesString.split(","))))
    attributeArray = attributeArray.reshape((32, 84))
    return attributeArray, [animal.strip("\t\'") for animal in names]


def importTrainingBallisticData():
    attributesString = open("data_lab2/ballist.dat", 'r').read()
    trainingPairStringList = attributesString.split("\n")
    trainingDataset = []
    for i in range(len(trainingPairStringList) - 1):
        trainingPair = trainingPairStringList[i].split("\t")
        # print(trainingPair)
        input = tuple(map(float, trainingPair[0].split(" ")))
        output = tuple(map(float, trainingPair[1].split(" ")))

        trainingDataset.append((input, output))
        # print(input, output)

    return np.array(trainingDataset) #Returns an array with a tuple containing 2 elements:  1) input tuple 2) output tuple (see instructions)


def importTestBallisticData():
    attributesString = open("data_lab2/balltest.dat", 'r').read()
    testPairStringList = attributesString.split("\n")
    testDataset = []
    for i in range(len(testPairStringList) - 1):
        trainingPair = testPairStringList[i].split("\t")
        # print(trainingPair)
        input = tuple(map(float, trainingPair[0].split(" ")))

        testDataset.append(input)
        # print(input, output)

    return np.array(testDataset) #Returns an array with a input tuples


if __name__ == '__main__':
    print("TRAIN")
    print(importTrainingBallisticData()[4])
    print("TEST")
    print(importTestBallisticData()[10])

    # a, b = importAnimalDataSet()
    # print(b)

    # print(trainingDataSet)
    # print(trainingDataSet[0][0][0])
