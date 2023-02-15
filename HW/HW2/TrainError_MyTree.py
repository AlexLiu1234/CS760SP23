import math
from math import log
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt


def error_train(tree, test_set):
    y_test = np.array([predict_tree(x, tree) for x in test_set])
    class_labels_test = [int(data[-1]) for data in test_set]
    error_num = 0.0

    for num in range(len(test_set)):
      if y_test[num] != class_labels_test[num]:
        error_num += 1.0
    error_train = error_num/len(y_test)
    return error_train


if __name__ == '__main__':
    dataSet, label = LoadData("Dbig.txt")
    random.shuffle(dataSet)

    train_set = dataSet[:8192]
    test_set = dataSet[8192:]

    random.shuffle(train_set)
    D_2048_train = train_set[:2048]

    random.shuffle(train_set)
    D_512_train = train_set[:512]

    random.shuffle(train_set)
    D_128_train = train_set[:128]

    random.shuffle(train_set)
    D_32_train = train_set[:32]

    ####################################
    #-- Train Error--#
    error_list = []

    tree = BuildDecisionTree_FOR_BOUNDARY(D_32_train, labels)
    error_32 = error_train(tree, test_set)
    error_list.append(error_32)

    tree = BuildDecisionTree_FOR_BOUNDARY(D_128_train, labels)
    error_128 = error_train(tree, test_set)
    error_list.append(error_128)

    tree = BuildDecisionTree_FOR_BOUNDARY(D_512_train, labels)
    error_512 = error_train(tree, test_set)
    error_list.append(error_512)

    tree = BuildDecisionTree_FOR_BOUNDARY(D_2048_train, labels)
    error_2048 = error_train(tree, test_set)
    error_list.append(error_2048)

    tree = BuildDecisionTree_FOR_BOUNDARY(train_set, labels)
    error_8192 = error_train(tree, test_set)
    error_list.append(error_8192)


    n = [32,128,512,2048,8192]

    plt.figure(figsize = (11,4))
    plt.title('Learning Curve')
    plt.plot(n,error_list)
    plt.xlabel('Size of DataSet')
    plt.ylabel('Error Rate')

    plt.show()