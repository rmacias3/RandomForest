import csv
from math import log
import numpy as np
import random as rd
from CSE6242HW4Tester import generateSubmissionFile

myname = "Macias-Ricardo"


def entropyMeasure(examples):
    length = len(examples)
    if length == 0:
        return 0
    posVals = 0
    for k in examples:
        if k[-1] == 1:
            posVals += 1
    posProb = float(posVals) / length
    negProb = float(length - posVals) / length
    if posProb != 0 and negProb != 0:
        return -(posProb * log(posProb, 2) + negProb * log(negProb, 2))
    else:
        return 0


"""
Given an index of an attribute and a value an attribute that you wish to split on,
returns two list of examples, a list where examples are have an attribute value less
than/equal to the one specified, and another one where it's greater.
"""


def split(examples, attr, attrVal):
    setOne = []
    setTwo = []
    for k in examples:
        if k[attr] <= attrVal:
            setOne.append(k)
        else:
            setTwo.append(k)
    return (np.array(setOne), np.array(setTwo))


"""
This method uses the one above repeatedly to iterate through all the possible splits of examples to find the two
most "separated lists" or the ones with most information gain. This is based off the formula in page 717 of the book as well.
"""


def pickBestPair(examples, length):
    maxFeat = ""
    maxIG = 0.0
    entropyOfAllExamples = entropyMeasure(examples)
    # examples = np.array(examples)
    # print "is this"
    for k in range(len(examples[0]) - 1):
        uniqueFeatureValues = set(examples[:, k].tolist())
        valuesToGoThrough = np.random.choice(list(uniqueFeatureValues), len(uniqueFeatureValues) / 2, replace=False)
        median = sorted(uniqueFeatureValues)[len(uniqueFeatureValues) / 2]
        if median not in valuesToGoThrough:
            np.append(valuesToGoThrough, median)
        # print "wut"
        for val in valuesToGoThrough:
            # print "start splitting"
            setOne, setTwo = split(examples, k, val)
            # Calculate info gain
            tempEntropy = (len(setOne) / float(length)) * entropyMeasure(setOne) + (len(setTwo) / float(
                length)) * entropyMeasure(setTwo)
            # print "finish Info gain"
            infoGain = entropyOfAllExamples - tempEntropy
            if (infoGain > maxIG):
                maxIG = infoGain
                maxFeat = k
                bestVal = val
    # print "expensive"
    return maxFeat, bestVal


# Less efficient thatn list slicing for checking if all outputs are same
def checkEquality(outputs):
    first = outputs[0]
    for k in outputs:
        if k != first:
            return False
    return True


# Did not help very much, but it was just classifing an example by majority
def majorityRulesByLargeMargin(count, output, length):
    if float(count) / length >= .92:
        return True, output
    elif float(count) / length <= .08:
        if int(output) == 1:
            return True, 0.0
        else:
            return True, 1.0
    else:
        return False, None


"""
Creator of decision tree. uses return values of methods above to subsequently
build more trees of of it. returns it in the form of a dictionary
"""


def build_tree(examples, level, maxLevel):
    level += 1
    outputs = [k[-1] for k in examples]
    countOfOneOrTheOther = outputs.count(outputs[0])
    length = len(outputs)
    if length == countOfOneOrTheOther:
        return outputs[0]
        # boolean, classify = majorityRulesByLargeMargin(countOfOneOrTheOther, outputs[0], length)
        # if boolean:
        # return classify
    if level >= maxLevel:
        if countOfOneOrTheOther >= len(outputs) / 2:
            return outputs[0]
        else:
            if outputs[0] == 1.0:
                return 0.0
            else:
                return 1.0
    maxFeat, bestVal = pickBestPair(examples, len(examples))
    setOne, setTwo = split(examples, maxFeat, bestVal)
    tree = {maxFeat: {}}
    tree[maxFeat][">" + str(bestVal)] = build_tree(setTwo, level, maxLevel)
    tree[maxFeat]["<=" + str(bestVal)] = build_tree(setOne, level, maxLevel)
    return tree


class RandomForest(object):
    class __DecisionTree(object):
        def __init__(self):
            self.tree = ""

        def learn(self, examples, tree_depth):
            self.tree = build_tree(examples, 0, tree_depth)

        def classify(self, test_instance):
            tree = self.tree
            # Iterates through the built tree to classify value
            while (tree != 0 and tree != 1):
                attribute = tree.keys()[0]
                value = float(test_instance[attribute])
                tree = tree[attribute]
                splitter = tree.keys()[0]
                if ">" in splitter:
                    # ">number"
                    myVal = float(splitter[1:])
                    if value <= myVal:
                        tree = tree[tree.keys()[1]]
                    else:
                        tree = tree[tree.keys()[0]]
                else:
                    # "<=number"
                    myVal = float(splitter[2:])
                    if value <= myVal:
                        tree = tree[tree.keys()[0]]
                    else:
                        tree = tree[tree.keys()[1]]
            return int(tree)

    decision_trees = []

    # this generates examples with a random set of attributes drawn
    def generateExamples(self, X, Y, numAttributes):
        # print "generate"
        examples = []
        for i in range(len(X)):
            k = np.append(X[i], Y[i])
            examples.append(k.tolist())
        exTemp = np.array(examples)
        if numAttributes == 11:
            return exTemp
        attributesOne = set([i for i in range(len(examples[0]) - 1)])
        print
        attributesOne
        attributesTwo = set((np.random.choice(list(attributesOne), numAttributes, replace=False)).tolist())
        # print attributesTwo
        removeThese = list(attributesOne.difference(attributesTwo))
        # print removeThese
        hello = np.delete(exTemp, removeThese, axis=1)
        # print "finish generating"
        return hello

    def __init__(self, num_trees, subset_size, tree_depth):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees
        self.tree_depth = tree_depth
        self.subset_size = subset_size

    # You MUST NOT change this signature

    def fit(self, X, y):
        # TODO: train `num_trees` decision trees
        for k in self.decision_trees:
            thing = self.generateExamples(X, y, self.subset_size)
            k.learn(thing, self.tree_depth)

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype=int)
        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])
            counts = np.bincount(votes)
            y = np.append(y, np.argmax(counts))
        return y


def main():
    X = []
    y = []
    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Split training/test sets
    # You need to modify the following code for cross validation
    # K-Fold cross validaton
    K = 10
    allAccuracies = []
    for i in range(K):
        # Grab a random set of 3600 rows to train on
        rowsToTrain = np.random.randint(4000, size=3600)
        rowsToTest = []
        length = 0
        # Grab 400 rows that aren't in the training set to test
        for t in range(4002):
            if length == 400:
                break
            if t not in rowsToTrain:
                length += 1
                rowsToTest.append(t)
        X_train = X[rowsToTrain, :]
        X_test = X[rowsToTest, :]
        y_train = y[rowsToTrain]
        y_test = y[rowsToTest]
        # Parameters: #Num of trees, #attributes, max depth of tree
        randomForest = RandomForest(1, 11, 7)
        randomForest.fit(X_train, y_train)
        y_predicted = randomForest.predict(X_test)
        results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
        curAccuracy = float(results.count(True)) / float(len(results))
        print
        "Accuracy of tree ", str(i), ": ", curAccuracy
        # print  " Trees, ", " Attributes, ", curAccuracy, " Accuracy"
        allAccuracies.append(curAccuracy)
    total = sum(allAccuracies) / float(len(allAccuracies))
    print
    "accuracy: %.4f" % total
    generateSubmissionFile(myname, randomForest)
main()