import csv
import math
import random

#Read the data file
def loadCsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    dataset.pop(0)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

#Split the data set
def splitDataset(dataset,splitRatio):
    trainSize =  int(len(dataset)*splitRatio)
    trainDataset=[]
    copy = list(dataset)
    while(len(trainDataset)<trainSize):
        index = random.randrange(len(copy)) #use the first value to calculeate the next value until the end of the values
        trainDataset.append(copy.pop(index))
    return(trainDataset,copy)

def separateBbyClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if(vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarise(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)] #zip is for parallel iteration
    del summaries[-1]
    return summaries

def sumariseByClass(dataset):
    separeted = separateBbyClass(dataset)
    summaries = {}
    for classValue, instances in separeted.items():
        summaries[classValue] = summarise(instances)
    return summaries

def calculateProbability(x,mean,stdev):
    exponent  = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries,inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue]=1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue]*= calculateProbability(x, mean, stdev)
        return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries,testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

def main():
    filename = "diabetes.csv"
    splitRatio = 0.67
    
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset,splitRatio)

    print("Split {0} rows into train = {1} and test = {2} rows".format(len(dataset),len(trainingSet),len(testSet)))
    
    #preparate model
    summaries = sumariseByClass(trainingSet)

    #test model
    predictions = getPredictions(summaries,testSet)
    accuracy = getAccuracy(testSet,predictions)
    print("Accuracy: {0}%".format(accuracy))

main()