import collections
import io
import numpy as np
import os 
import re

#Relative Paths to Train and Test files.
dirPath = os.path.dirname(__file__)
trainPathHam = os.path.join(dirPath, 'train/ham')
trainPathSpam = os.path.join(dirPath, 'train/spam')

testPathHam = os.path.join(dirPath, 'test/ham')
testPathSpam = os.path.join(dirPath, 'test/spam')

#Set to 1 to remove stop words.
setStopWords = 1

#Default English Stop Words list from https://www.ranks.nl/stopwords
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

def removeStopWords():
    for i in stopWords:
        if i in hamDict:
            removedWord = hamDict.pop(i)
            print("Removed from Ham: " + i)
        if i in spamDict:
            removedWord = spamDict.pop(i)
            print("Removed from Spam: " + i)
        if i in combinedDict:
            removedWord = combinedDict.pop(i)
            print("Removed Stop Word: " + i)  

#Function to read file. Encoding should be changed based on files.
def readFile(fileName, filePath):
    with io.open(filePath + '\\' + fileName, 'r', encoding = 'iso-8859-1') as file:
        words = re.findall('[A-Za-z0-9\']+', file.read())
    lowerCaseWords = [i.lower() for i in words]  
    return lowerCaseWords
    
def getWords(filePath):
    fileCount = 0
    wordsList = list()
    for i in os.listdir(filePath):
        if i.endswith(".txt"):
            wordsList += readFile(i, filePath)
            fileCount += 1
    return wordsList, fileCount
    
def updateCount():
    for i in combinedDict:
        if i not in hamDict:
            hamDict[i] = 0
        if i not in spamDict:
            spamDict[i] = 0

hamWordList, hamCount = getWords(trainPathHam)
spamWordList, spamCount = getWords(trainPathSpam)

def probabilityOf(classifier):
    if classifier == "spam":
        probabilityOfSpam = spamCount/(spamCount + hamCount)
        return probabilityOfSpam
    else:
        probabilityOfHam = hamCount/(spamCount + hamCount)
        return probabilityOfHam

hamDict = dict(collections.Counter(i.lower() for i in hamWordList))
spamDict = dict(collections.Counter(i.lower() for i in spamWordList))
combinedWordList = hamWordList + spamWordList
combinedDict = collections.Counter(combinedWordList)

updateCount()

probabilityOfHamWord = dict()
probabilityOfSpamWord = dict()

if setStopWords == 1:
    removeStopWords()

def probabilityOfClassifier(classifier):
    i = 0                   
    if classifier == "ham":
        for word in hamDict:
            i += (hamDict[word] + 1)
        for word in hamDict:
            probabilityOfHamWord[word] = np.log2((hamDict[word] + 1)/i)
    elif classifier == "spam":
        for word in spamDict:
            i += (spamDict[word] + 1)
        for word in spamDict:
            probabilityOfSpamWord[word] = np.log2((spamDict[word] + 1)/i) 
           
probabilityOfClassifier("ham")
probabilityOfClassifier("spam") 
   
def classify(filePath, classifier):
    wrongClassifications = 0
    fileCount = 0
    for fileName in os.listdir(filePath):
            words = readFile(fileName, filePath)
            hamProbability = np.log2(probabilityOf("ham"))
            spamProbability = np.log2(probabilityOf("spam"))
            for word in words:
                if word in probabilityOfHamWord:
                    hamProbability += probabilityOfHamWord[word]
                if word in probabilityOfSpamWord:
                    spamProbability += probabilityOfSpamWord[word]
            fileCount += 1
            if classifier == "spam":
                if hamProbability >= spamProbability:
                    wrongClassifications += 1
            elif classifier == "ham":
                if hamProbability <= spamProbability:
                    wrongClassifications += 1
    return wrongClassifications, fileCount
    
if setStopWords:
    print("\nNaive Bayes Classification With Stop Words: ")  
else:
    print("\nNaive Bayes Classification Without Stop Words: ")  
     
wrongHamClassification, hamFileCount = classify(testPathHam, "ham")
wrongSpamClassification, spamFileCount = classify(testPathSpam,"spam")
hamAccuracy = round(((hamFileCount - wrongHamClassification )/(hamFileCount ))*100, 2)
spamAccuracy = round(((spamFileCount -  wrongSpamClassification )/(spamFileCount))*100, 2)
totalFileCount = hamFileCount + spamFileCount
totalWrongClassifications = wrongHamClassification + wrongSpamClassification
accuracy = round(((totalFileCount  - totalWrongClassifications )/totalFileCount)*100,2)

print("\nHam File Count: ", hamFileCount)
print("\nSpam File Count: ", spamFileCount)
print("\nTotal File Count: ", totalFileCount)
print("\nCount of files classified as Ham: ", hamFileCount - wrongHamClassification)
print("\nCount of files wrongly classified as Ham: ", wrongHamClassification)
print("\nAccuracy for Ham Classification: " + str(hamAccuracy) + "%")
print("\nCount of files classified as Spam: ", spamFileCount - wrongSpamClassification)
print("\nCount of files wrongly classified as Spam:", wrongSpamClassification)
print("\nAccuracy For Spam Classification: " + str(spamAccuracy) + "%") 
print("\nOverall Accuracy: " + str(accuracy) + "%" + "\n")