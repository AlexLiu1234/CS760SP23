import math
from math import log
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt


def LoadData(filename):
    dataSet = []
    with open(filename,encoding='utf-8') as f:
        for line in f:
            x1,x2,y=line.strip().split(" ")
            dataSet.append([x1,x2,y])
    f.close()
    labels = ['X1','X2']

    return dataSet, labels


def CalculateEntropy(dataSet,index): 
    entropy = 0.0
    targetSet = [data[index] for data in dataSet] 
    instances = set(targetSet)
    
    for value in instances:
        prob = targetSet.count(value)/len(dataSet)
        entropy -= prob * log(prob, 2)
        
    return entropy


def DetermineCandidateSplits(dataSet, index):
    sortedData = sorted(dataSet, key=itemgetter(index))
    return [data[index] for i, data in enumerate(sortedData[1:]) if data[-1] != sortedData[i][-1]]


def FindBestSplit(data_set):
    num_features = len(data_set[0]) - 1
    entropy = CalculateEntropy(data_set, -1)
    best_info_gain_ratio = 0.0
    best_feature_index = -1
    best_value = 0.0

    for i in range(num_features):
        candidate_splits = DetermineCandidateSplits(data_set, i)
        for value in candidate_splits:
            branch1 = []
            branch2 = []
            for feature in data_set:
                if feature[i] >= value:
                    branch1.append(feature)
                else:
                    branch2.append(feature)

            if branch1 and branch2:
                prob1 = len(branch1) / len(data_set)
                prob2 = len(branch2) / len(data_set)
                conditional_entropy = prob1 * CalculateEntropy(branch1, -1) + prob2 * CalculateEntropy(branch2, -1)
                entropy_value = -prob1 * math.log2(prob1) - prob2 * math.log2(prob2)
                info_gain = entropy - conditional_entropy


                if entropy_value:
                    info_gain_ratio = info_gain / entropy_value

                    if info_gain_ratio > best_info_gain_ratio:
                        best_info_gain_ratio = info_gain_ratio
                        best_feature_index = i
                        best_value = value

    return best_feature_index, best_value


def BuildDecisionTree_FOR_BOUNDARY(dataSet, labels):
    bestFeature, bestValue = FindBestSplit(dataSet) 
    classList = [data[-1] for data in dataSet] 

    if len(dataSet) == 0:
        return
    elif bestFeature == -1:
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys(): 
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
        if len(sortedClassCount) == 2 and sortedClassCount[0][1] == sortedClassCount[1][1]:
            return 1
        else:
            return int(sortedClassCount[0][0])
    
    bestFeatureLabel = labels[bestFeature]
    myTree = {'feature value': int(bestFeature), 'cuts': float(bestValue), 'left tree': {
        
    }, 'right tree': {}}

    leftDataSet = [data for data in dataSet if data[bestFeature] < bestValue]
    rightDataSet = [data for data in dataSet if data[bestFeature] >= bestValue]

    myTree['left tree'] = BuildDecisionTree_FOR_BOUNDARY(leftDataSet, labels)
    myTree['right tree'] = BuildDecisionTree_FOR_BOUNDARY(rightDataSet, labels)
    return myTree


def predict_tree(x, tree):
    while isinstance(tree, dict):
        if x[tree['feature value']] <= tree['cuts']:
            tree = tree['left tree']
        else:
            tree = tree['right tree']
    return tree


if __name__ == '__main__':
    # Load dataset
    dataSet, labels = LoadData('D2.txt')
    tree = BuildDecisionTree_FOR_BOUNDARY(dataSet, labels)

    # Extract features and labels from data set
    features = np.array(dataSet)[:, :2].astype(float)
    labels = np.array(dataSet)[:, 2]

    # Map label values to colors
    label_colors = np.where(labels == '0', 'blue', 'red')

    # Define meshgrid for visualization
    x0, x1 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X = np.c_[x0.ravel(), x1.ravel()]

    # Use decision tree to make predictions on meshgrid
    y = np.array([predict_tree(x, tree) for x in X])

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.title('Scatter Plot D_2')
    plt.contourf(x0, x1, y.reshape(x0.shape), cmap='Pastel2', alpha=0.9)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(features[:, 0], features[:, 1], alpha=0.5, c=label_colors)
    plt.show()
