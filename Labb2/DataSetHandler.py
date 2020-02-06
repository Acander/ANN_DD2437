import numpy as np
import string


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

    return np.array(
        trainingDataset)  # Returns an array with a tuple containing 2 elements:  1) input tuple 2) output tuple (see instructions)


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

    return np.array(testDataset)  # Returns an array with a input tuples


def importCities():
    attributesString = open("data_lab2/cities.dat", 'r').read()
    list = [element.strip("\n") for element in attributesString.split(";")]
    list.pop()
    list = np.array([tuple(map(float, t.split(","))) for t in list])

    return list  # Returns a list (numpy array) of tuples containing coordinates


def importVotesAndMPs():
    numMPs = 349

    # Import the 31 votes for each mp
    voteData = open("data_lab2/votes.dat", 'r').read()
    votes = np.reshape(list(map(float, voteData.split(","))), (349, 31))

    '''Import party:
    Coding: 0 = no party, 1 = 'm', 2 = 'fp', 3 = 's', 4 = 'v', 5 = 'mp', 6 = 'kd', 7 = 'c'
    Use some color scheme for these different groups
    Import sex and district:
    '''
    partyData = open("data_lab2/mpparty.dat", 'r').read().split("\n")
    partyData.pop()
    partyMp = [int(party.strip("\t ")) for party in partyData]

    # Import sex (Coding: Male 0, Female 1):
    mpSexData = open("data_lab2/mpsex.dat", 'r').read().split("\n")
    mpSexData.pop()
    mpSex = [int(mpSex.strip("\t ")) for mpSex in mpSexData]

    # Import district:
    mpDistrictData = open("data_lab2/mpdistrict.dat", 'r').read().split("\n")
    mpDistrictData.pop()
    mpDistrict = [int(mpDistrict.strip("\t ")) for mpDistrict in mpDistrictData]

    # Import names:
    mpNameData = open("data_lab2/mpnames.txt", 'r').read().split("\n")

    FinalMPInfoList = []
    for mp in range(numMPs):
        color, letter = colorAndLetterScheme(partyMp[mp])
        sex = parseSex(mpSex[mp])
        FinalMPInfoList.append((mpNameData[mp], sex, mpDistrict[mp], (partyMp[mp], letter, color)))

    return votes, FinalMPInfoList

    # Votes: 349X31 matrix with votes for each mp
    # FinalMPInfoList: List of tuples containing mp info
    # 1) Name 2) Sex 3) District 4) Tuple containing party
    # (along with name and color for that party)


def parseSex(sex):
    if sex == 0:
        return "Male"
    else:
        return "Female"


def colorAndLetterScheme(party):
    color = ""
    letter = ""
    if party == 1:
        color = "blue"
        letter = "m"
    elif party == 2:
        color = "aqua"
        letter = "fp"
    elif party == 3:
        color = "red"
        letter = "s"
    elif party == 4:
        color = "purple"
        letter = "v"
    elif party == 5:
        color = "green"
        letter = "mp"
    elif party == 6:
        color = "Teal"
        letter = "kd"
    elif party == 7:
        color = "Lime"
        letter = "c"
    else:
        color = "white"
        letter = "no party"
    return color, letter


if __name__ == '__main__':
    """print("TRAIN")
    print(importTrainingBallisticData()[4])
    print("TEST")
    print(importTestBallisticData()[10])"""

    # a, b = importAnimalDataSet()
    # print(b)

    # print(trainingDataSet)
    # print(trainingDataSet[0][0][0])

    # importCities()
    votes, mpInfoList = importVotesAndMPs()
    # print(mpInfoList[3])
    print(votes[5])
    print(mpInfoList[5])
