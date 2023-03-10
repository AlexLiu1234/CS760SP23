import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv('emails.csv', error_bad_lines=False)

#Accuracy = np.where(df >= 3000, 1, 0)
y = np.array(df.iloc[:, -1])
X = df.drop(columns=['Email No.','Prediction'])
folds = [
    (np.arange(0, 1000), np.arange(1000, 5000)),
    (np.arange(1000, 2000), np.concatenate([np.arange(0, 1000), np.arange(2000, 5000)])),
    (np.arange(2000, 3000), np.concatenate([np.arange(0, 2000), np.arange(3000, 5000)])),
    (np.arange(3000, 4000), np.concatenate([np.arange(0, 3000), np.arange(4000, 5000)])),
    (np.arange(4000, 5000), np.arange(0, 4000)),]

# Perform cross-validation using the defined folds
K=[]
K_num=[1,3,5,7,10]
K_avg = []

for num in K_num:
  for i, (test_idx, train_idx) in enumerate(folds):
      X_train, X_test = X.iloc[train_idx, :-1], X.iloc[test_idx, :-1]
      y_train, y_test = y[train_idx], y[test_idx]
      # Create KNN classifier
      knn = KNeighborsClassifier(n_neighbors = num)
      # Fit the classifier to the data
      knn.fit(X_train, y_train)
      K.append(knn.score(X_test, y_test))

  K_avg.append(np.mean(K))
  K=[]

K_avg

for i in range(5):
  print(f'K= {K_num[i]}: Average accuracy is {K_avg[i]:0.4}')

plt.plot(K_num, K_avg)
plt.scatter(K_num, K_avg)
plt.xlabel('K')
plt.ylabel('Average accuracy')
plt.title('KNN 5-Fold Cross Validation')
plt.show()

K = []
y_pred=[]

for i, (test_idx, train_idx) in enumerate(folds):
    X_train, X_test = X.iloc[train_idx, :-1], X.iloc[test_idx, :-1]
    y_train, y_test = y[train_idx], y[test_idx]
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 1)
    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    Recall = recall_score(y_test, y_pred)
    
    print(f'Fold {i+1}: Accuracy is {Accuracy:0.4}, Precision is {Precision:0.4}, Recall is {Recall:0.4}')

# Load the dataset
data = pd.read_csv('emails.csv')
X_train = data.iloc[:4000, 1:-1]
y_train = data.iloc[:4000, -1]
X_test = data.iloc[4000:, 1:-1]
y_test = data.iloc[4000:, -1]

# Train kNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Calculate predicted probabilities for kNN
y_hat_knn = knn.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and threshold for kNN
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, y_hat_knn)

# Calculate the AUC (area under the curve) for kNN
knn_auc = roc_auc_score(y_test, y_hat_knn)

# Plot the ROC curves
plt.plot(knn_fpr, knn_tpr, label='kNN (AUC = {:.2f})'.format(knn_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for kNN')
plt.legend()
plt.show()
