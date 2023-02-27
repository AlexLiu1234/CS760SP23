import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric
from matplotlib.patches import Patch

def load_data(datafile):
    data = np.genfromtxt(datafile)
    X = []
    Y = []
    for row in data:
        X.append(row[:-1])
        Y.append(row[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y, data

X, Y, _ = load_data('D2z.txt')

Y_str = []
for i in range(len(Y)):
    if Y[i] == 0.:
        Y_str.append('o')
    else:
        Y_str.append('+')

X_test = []
for i in np.arange(-2,2,0.1):
    for j in np.arange(-2,2,0.1):
        X_test.append([i,j])
        
X_test = np.array(X_test)

# dist = DistanceMetric.get_metric('euclidean')
clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(X,Y)
test_labels = clf.predict(X_test)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'skyblue'])
cmap_bold = ListedColormap(['blue', 'c', 'red'])

xx,yy = np.meshgrid(np.arange(-3,3,0.2), np.arange(-3,3,0.2))
xx_labels = clf.predict(np.c_[xx.ravel(), yy.ravel()])
xx_labels = xx_labels.reshape(xx.shape)
test_labels = clf.predict(X_test)

plt.figure(figsize=(10,10))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels.ravel(), cmap=cmap_bold, s=8)
plt.scatter(X[:,0], X[:,1], c = Y)
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# True labels of the data
y_true = ['+','+','-','+','+', '-', '+','+', '-', '-']

# Predicted probabilities of the positive class
y_score = [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]

y_score = np.array(y_score)
y_score = np.clip(y_score, 0, 1)

# Compute the false positive rate (fpr), true positive rate (tpr),
# and threshold values for the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label='+')

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Annotate the ROC curve with the threshold values
points = list(zip(fpr, tpr, thresholds))
for point in points:
    if point[2] < 1:
        plt.annotate('{:.2f}'.format(point[2]), xy=point[:2], textcoords='offset points', xytext=(0,10), ha='center')

plt.xlabel('False Positive Rate (positive label: +)')
plt.ylabel('True Positive Rate (positive label: +)')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()
