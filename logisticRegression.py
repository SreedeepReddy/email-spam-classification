import collections
import io
import numpy as np
import os
import re

stopWords = ['a'
,'about'
,'above'
,'after'
,'again'
,'against'
,'all'
,'am'
,'an'
,'and'
,'any'
,'are'
,'aren\'t'
,'as'
,'at'
,'be'
,'because'
,'been'
,'before'
,'being'
,'below'
,'between'
,'both'
,'but'
,'by'
,'can\'t'
,'cannot'
,'could'
,'couldn\'t'
,'did'
,'didn\'t'
,'do'
,'does'
,'doesn\'t'
,'doing'
,'don\'t'
,'down'
,'during'
,'each'
,'few'
,'for'
,'from'
,'further'
,'had'
,'hadn\'t'
,'has'
,'hasn\'t'
,'have'
,'haven\'t'
,'having'
,'he'
,'he\'d'
,'he\'ll'
,'he\'s'
,'her'
,'here'
,'here\'s'
,'hers'
,'herself'
,'him'
,'himself'
,'his'
,'how'
,'how\'s'
,'i'
,'i\'d'
,'i\'ll'
,'i\'m'
,'i\'ve'
,'if'
,'in'
,'into'
,'is'
,'isn\'t'
,'it'
,'it\'s'
,'its'
,'itself'
,'let\'s'
,'me'
,'more'
,'most'
,'mustn\'t'
,'my'
,'myself'
,'no'
,'nor'
,'not'
,'of'
,'off'
,'on'
,'once'
,'only'
,'or'
,'other'
,'ought'
,'our'
,'ours']

setStopWords = 1

lambdaVal = 10
iterationLimit = 50
learningRate = 0.001

dirPath = os.path.dirname(__file__)
trainPathHam = os.path.join(dirPath, 'train/ham')
trainPathSpam = os.path.join(dirPath, 'train/spam')

testPathHam = os.path.join(dirPath, 'test/ham')
testPathSpam = os.path.join(dirPath, 'test/spam')

def readFile(fileName, filePath):
    with io.open(filePath + '\\' + fileName, 'r', encoding = 'iso-8859-1') as file:
        words = re.findall('[A-Za-z0-9\']+', file.read())
    lowerCaseWords = [i.lower() for i in words]  
    return lowerCaseWords

def getWords(filePath):
    wordList = list()
    fileCount = 0
    for i in os.listdir(filePath):
        if i.endswith(".txt"):
            wordList += readFile(i, filePath)
            fileCount += 1
    return wordList, fileCount
    
def removeStopWords():
    for i in stopWords:
        if i in hamTrain:
            hamTrain.remove(i)
        if i in spamTrain:
            spamTrain.remove(i)
        if i in hamTest:
            hamTest.remove(i)
        if i in spamTest:
            spamTest .remove(i)

def createMatrix(row, column):
    return np.zeros((row, column))
hamTrain, hamTrainCount = getWords(trainPathHam)
spamTrain, spamTrainCount = getWords(trainPathSpam)

hamTest, hamTestCount = getWords(testPathHam)
spamTest, spamTestCount = getWords(testPathSpam)

if setStopWords == 1:
    removeStopWords()

totalFiles = hamTrainCount + spamTrainCount
combinedTrain = hamTrain + spamTrain
combinedTrainDict = collections.Counter(combinedTrain)
combinedListTrain = list(combinedTrainDict.keys())
classification = list()  
combinedTest = hamTest + spamTest
combinedTestDict = collections.Counter(combinedTest)
combinedTestList = list(combinedTestDict.keys())
totalTestFiles = hamTestCount + spamTestCount

trainMatrix = createMatrix(totalFiles, len(combinedListTrain))
testMatrix = createMatrix(totalTestFiles, len(combinedTestList))
def progress(fileNumber):
    progressPercent = round((fileNumber/totalTestFiles)*100, 2)
    print("Progress: " + str(progressPercent) + "%")
    

def initializeLists():
    sigmoidL = list() 
    testClassification = list() 
    weightOfFeature = list()   
    for i in range(totalFiles):
        sigmoidL.insert(len(sigmoidL), -1)
        classification.insert(len(classification), -1)    
    for i in range(totalTestFiles):
        testClassification.insert(len(testClassification), -1)    
    for i in range(len(combinedListTrain)):
        weightOfFeature.insert(len(weightOfFeature), 0)
    return sigmoidL, classification, testClassification, weightOfFeature

sigmoidL, classification, testClassification, weightOfFeature = initializeLists()
setRow = 0

def makeMatrix(featureMatrix, filePath, combinedListTrain, setRow, classifier, classification):
    for fileName in os.listdir(filePath):
        words = readFile(fileName, filePath)
        i = dict(collections.Counter(words))
        for key in i:
            if key in combinedListTrain:
                setColumn = combinedListTrain.index(key)
                featureMatrix[setRow][setColumn] = i[key]
        if (classifier == "ham"):
            classification[setRow] = 0
        elif (classifier == "spam"):
            classification[setRow] = 1
        setRow += 1
    return featureMatrix, setRow, classification

testRowMatrix = 0

trainMatrix, setRow, classification = makeMatrix(trainMatrix, trainPathHam, combinedListTrain, setRow, "ham", classification)
trainMatrix, setRow, classification = makeMatrix(trainMatrix, trainPathSpam, combinedListTrain, setRow, "spam", classification)

testMatrix, testRowMatrix, testClassification = makeMatrix(testMatrix, testPathHam, combinedTestList, testRowMatrix, "ham", testClassification)
testMatrix, testRowMatrix, testClassification = makeMatrix(testMatrix, testPathSpam, combinedTestList, testRowMatrix, "spam", testClassification)

def sigmoid(Val):
    return 1.0 / (1 + np.exp(-Val))

def trainingFunction(totalFiles, numbeOffeatures, featureMatrix, classification):
    global sigmoid
    global sigmoidL
    for files in range(totalFiles):
        summation = 1.0
        for features in range(numbeOffeatures):
            summation += featureMatrix[files][features] * weightOfFeature[features]
        sigmoidL[files] = sigmoid(summation)
    for feature in range(numbeOffeatures):
        weight = 0
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            classifier = classification[files]
            sigValue = sigmoidL[files]
            weight += frequency * (classifier - sigValue)
        prevWeight = weightOfFeature[feature]
        weightOfFeature[feature] += ((weight * learningRate) - (learningRate * lambdaVal * prevWeight))
    return weightOfFeature

def train():
    for i in range(int(iterationLimit)):
        trainingFunction(totalFiles, len(combinedListTrain), trainMatrix, classification)

def classify():
    correctHam = 0
    incorrectHam = 0
    correctSpam = 0
    incorrectSpam = 0
    ind=0
    for j in range(totalTestFiles):
        progress(ind+1)
        sumS = 1.0
        for i in range(len(combinedTestList)):
            word = combinedTestList[i]

            if word in combinedListTrain:
                index = combinedListTrain.index(word)
                weight = weightOfFeature[index]
                wordcount = testMatrix[j][i]

                sumS += weight * wordcount

        sig = sigmoid(sumS)
        if (testClassification[j] == 0):
            if sig < 0.5:
                correctHam += 1.0
            else:
                incorrectHam += 1.0
        else:
            if sig >= 0.5:
                correctSpam += 1.0
            else:
                incorrectSpam += 1.0
        ind += 1
      
    print("\nHam File Count: " + str(hamTestCount))
    print("\nSpam File Count: " + str(spamTestCount)) 
    print("\nTotal File Count: " + str(totalTestFiles)) 
    print("\nAccuracy for Ham Classification: " + str(round(((correctHam / (correctHam + incorrectHam)) * 100), 2)))
    print("\nAccuracy for Spam Classification: " + str(round(((correctSpam / (correctSpam + incorrectSpam)) * 100), 2)))
    print("\nOverall Accuracy: " + str(round((((correctHam + correctSpam) / (correctHam + incorrectHam+correctSpam + incorrectSpam)) * 100), 2)))

print("\nTraining...")
train()
print("\nTraining done.\nClassifying...")
classify()
