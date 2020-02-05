import numpy as np

def importDataSeta():
    # namesArray = np.array(list(namesString.split("\n")))
    # namesString = open("data_lab2/animalnames.txt", 'r').read()
    # namesArray = np.array(list(namesString.split("\n")))

    attributesString = open("data_lab2/animals.dat", 'r').read()
    #print(attributesString)

    attributeArray = np.array(list(map(int, attributesString.split(","))))
    attributeArray = attributeArray.reshape((32, 84))

    #print(attributeArray)

    return attributeArray