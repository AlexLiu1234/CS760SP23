import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv('emails.csv', error_bad_lines=False)

class LogisticRegression:
    def __init__(self, lr=0.1, num_iter=12000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __sigmoid(self, z):
        # Prevent overflow problem
        h = np.zeros(len(z))
        for i in range(len(z)):
            if z[i]<0: 
                h[i] = np.exp(z[i]) / (1 + np.exp(z[i]))
            else: 
                h[i] = 1 / (1 + np.exp(-z[i]))
        return h
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __loss(self, h, y, eps=1e-15):
        # clip predicted probability to avoid taking the logarithm of 0
        h = np.clip(h, eps, 1 - eps)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if self.verbose and i % 1000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'Loss: {self.__loss(h, y)} \t')
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

# Prepare features and labels
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Initialize logistic regression model
lr = LogisticRegression(lr=0.01, num_iter=10000, verbose=True)

# Initialize the KFold cross validator
kf = KFold(n_splits=5, shuffle=False, random_state= None)

# Initialize lists to store the evaluation metrics
accuracies = []
precisions = []
recalls = []

# the splits of datasets:
# – Fold 1, test set: Email 1-1000, training set: the rest (Email 1001-5000)
# – Fold 2, test set: Email 1000-2000, training set: the rest
# – Fold 3, test set: Email 2000-3000, training set: the rest
# – Fold 4, test set: Email 3000-4000, training set: the rest
# – Fold 5, test set: Email 4000-5000, training set: the rest

# Perform 5-fold cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    # Split the data into training and validation sets for this fold
    X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
    X_fold_val, y_fold_val = X.iloc[test_idx], y.iloc[test_idx]
    
    # Fit logistic regression model
    lr.fit(X_fold_train, y_fold_train)

    # Evaluate the logistic regression model on the validation set for this fold
    y_pred_val = lr.predict(X_fold_val)

    # Calculate the evaluation metrics for this fold
    accuracy = accuracy_score(y_fold_val, y_pred_val)
    precision = precision_score(y_fold_val, y_pred_val)
    recall = recall_score(y_fold_val, y_pred_val)

    # Print the evaluation metrics for this fold
    print(f"Fold {fold+1}: accuracy={accuracy}, precision={precision}, recall={recall}")

    # Add the evaluation metrics for this fold to the respective lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Calculate the average evaluation metrics over all folds
avg_accuracy = sum(accuracies) / len(accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)

# Print the average evaluation metrics over all folds
print(f"Average: accuracy={avg_accuracy}, precision={avg_precision}, recall={avg_recall}")

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

# Train logistic regression
lr = LogisticRegression(lr=0.01)
lr.fit(X_train, y_train)

# Calculate predicted probabilities for logistic regression
y_hat_lr = lr.predict_proba(X_test)

# Calculate the false positive rate, true positive rate, and threshold for Logistic regression
logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, y_hat_lr)

# Calculate the AUC (area under the curve) for logistic regression
logreg_auc = roc_auc_score(y_test, y_hat_lr)

# Plot the ROC curves
plt.plot(knn_fpr, knn_tpr, label='KNeighborsClassifier (AUC = {:.2f})'.format(knn_auc))
plt.plot(logreg_fpr, logreg_tpr, label='LogisticRegression (AUC = {:.2f})'.format(logreg_auc), color='orange')
plt.xlabel('False Positive Rate (Positive Label: 1)')
plt.ylabel('True Positive Rate (Positive Label: 1)')
plt.title('ROC Curves for Logistic Regression')
plt.legend()
plt.show()
