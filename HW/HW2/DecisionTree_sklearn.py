import math
from math import log
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def Test_MyTree_sklearn(trainSet, testSet):
    X_train = [[data[0], data[1]] for data in trainSet]
    y_train = [data[2] for data in trainSet]
    X_test = [[data[0], data[1]] for data in testSet]
    y_test = [data[2] for data in testSet]

    my_tree = DecisionTreeClassifier()
    my_tree.fit(X_train, y_train)

    y_pred = my_tree.predict(X_test)

    error_num = 0.0

    for num in range(len(y_test)):
      if y_test[num] != y_pred[num]:
        error_num += 1.0

    error_train = error_num/len(y_test)
    Node_num = my_tree.tree_.node_count

    return error_train, Node_num



if __name__ == '__main__':
    error_sklearn = []

    error_32, num_32 = Test_MyTree_sklearn(D_32_train,test_set)
    error_sklearn.append(error_32)
    error_128, num_128 = Test_MyTree_sklearn(D_128_train,test_set)
    error_sklearn.append(error_128)
    error_512, num_512 = Test_MyTree_sklearn(D_512_train,test_set)
    error_sklearn.append(error_512)
    error_2048, num_2048 = Test_MyTree_sklearn(D_2048_train,test_set)
    error_sklearn.append(error_2048)
    error_8192, num_8192 = Test_MyTree_sklearn(train_set,test_set)
    error_sklearn.append(error_8192)


    print("The number of node for D32 is", num_32, ",", "and the error is", error_32)
    print("The number of node for D128 is", num_128, ",", "and the error is", error_128)
    print("The number of node for D512 is", num_512, ",", "and the error is", error_512)
    print("The number of node for D2048 is", num_2048, ",", "and the error is", error_2048)
    print("The number of node for D8192 is", num_8192, ",", "and the error is", error_8192)


    n_sklearn = [32,128,512,2048,8192]

    plt.figure(figsize = (10,5))
    plt.title('Learning Curve By sklearn')
    plt.plot(n_sklearn, error_sklearn)
    plt.xlabel('Size of DataSet')
    plt.ylabel('Error Rate')

    plt.show()