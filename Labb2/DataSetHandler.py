import numpy as np

def importAnimalDataSet():
    namesString = open("data_lab2/animalnames.txt", 'r').read()
    names = np.array(list(namesString.split("\n")))

    attributesString = open("data_lab2/animals.dat", 'r').read()
    attributeArray = np.array(list(map(int, attributesString.split(","))))
    attributeArray = attributeArray.reshape((32, 84))
    return attributeArray, [animal.strip("\t\'") for animal in names]

def importBallisticDataSet():
    attributesString = open("data_lab2/ballist.dat", 'r').read()
    trainingPairStringList = attributesString.split("\n")
    trainingDataset = []
    for i in range(len(trainingPairStringList)-1):
        trainingPair = trainingPairStringList[i].split("\t")
        #print(trainingPair)
        input = tuple(map(float, trainingPair[0].split(" ")))
        output = tuple(map(float, trainingPair[1].split(" ")))

        trainingDataset.append((input, output))
        #print(input, output)

    return np.array(trainingDataset)

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

    return np.array(trainingDataset)

def importTestBallisticData():
    attributesString = open("data_lab2/balltest.dat", 'r').read()
    trainingPairStringList = attributesString.split("\n")
    trainingDataset = []
    for i in range(len(trainingPairStringList) - 1):
        trainingPair = trainingPairStringList[i].split("\t")
        # print(trainingPair)
        input = tuple(map(float, trainingPair[0].split(" ")))

        trainingDataset.append(input)
        # print(input, output)

    return np.array(trainingDataset)

trainingDataSet = importBallisticDataSet()
a, b = importAnimalDataSet()
print(b)

#print(trainingDataSet)
#print(trainingDataSet[0][0][0])