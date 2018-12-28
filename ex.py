# https://www.kaggle.com/clemensmzr/simple-multivariate-gaussian-anomaly-detection/notebook
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
data = pd.read_csv('creditcard.csv')

# # if you want to see th data or get some stats
# print(data.head())
# print(data.describe())

# get the normal and fraudulent cases
normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]

pca_columns = list(data)[1:-2]
# # visualize the distribution of the different features of the normal cases
# matplotlib.style.use('ggplot') # style used by some R package
# 
# normal_data[pca_columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2))
# plt.show()

# # visualize the distribution of the amount of data withdrawn
# normal_data["Amount"].loc[normal_data["Amount"] < 500].hist(bins=100)
# plt.show()


# see the mean and median of the normal and fraudulent datasets
# print("Mean", normal_data["Amount"].mean(), fraud_data["Amount"].mean())
# print("Median", normal_data["Amount"].median(), fraud_data["Amount"].median())


# analysis of the time of withdrawal
# normal_data["Time"].hist(bins=100)
# fraud_data["Time"].hist(bins=50)

# visualize correlations
normal_pca_data = normal_data[pca_columns]
fraud_pca_data = fraud_data[pca_columns]
plt.matshow(normal_pca_data.corr())
plt.show()


# Multivariate Gaussian analysis of the normal data

num_test = 75000
shuffled_data = normal_pca_data.sample(frac=1)[:-num_test].values # shuffle the dataset


# the fraud cases are distributed between the cross-validation and test cases
X_train = shuffled_data[:-2*num_test]

X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])
y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])

X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[246:]])
y_test = np.concatenate([np.zeros(num_test), np.ones(246)])


# build own covariance matrix fn, numpy will run into memory problems for large matrices
def covariance_matrix(X):
    m, n = X.shape 
    tmp_mat = np.zeros((n, n))
    mu = X.mean(axis=0)
    for i in range(m):
        tmp_mat += np.outer(X[i] - mu, X[i] - mu)
    return tmp_mat / m


cov_mat = covariance_matrix(X_train)

cov_mat_inv = np.linalg.pinv(cov_mat)
cov_mat_det = np.linalg.det(cov_mat)
def multi_gauss(x):
    n = len(cov_mat)
    return (np.exp(-0.5 * np.dot(x, np.dot(cov_mat_inv, x.T))) 
            / (2. * np.pi)**(n/2.) 
            / np.sqrt(cov_mat_det))

from sklearn.metrics import confusion_matrix

def stats(X_test, y_test, eps):
    predictions = np.array([multi_gauss(x) <= eps for x in X_test], dtype=bool)
    y_test = np.array(y_test, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    recall = tp / (tp + fn)
    prec = tp / (tp + fp)
    F1 = 2 * recall * prec / (recall + prec)
    return recall, prec, F1

eps = max([multi_gauss(x) for x in fraud_pca_data.values])
print(eps)


recall, prec, F1 = stats(X_valid, y_valid, eps)
print("For a boundary of:", eps)
print("Recall:", recall)
print("Precision:", prec)
print("F1-score:", F1)


validation = []
for thresh in np.array([1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]) * eps:
    recall, prec, F1 = stats(X_valid, y_valid, thresh)
    validation.append([thresh, recall, prec, F1])


x = np.array(validation)[:, 0]
y1 = np.array(validation)[:, 1]
y2 = np.array(validation)[:, 2]
y3 = np.array(validation)[:, 3]
plt.plot(x, y1)
plt.title("Recall")
plt.xscale('log')
plt.show()
plt.plot(x, y2)
plt.title("Precision")
plt.xscale('log')
plt.show()
plt.plot(x, y3)
plt.title("F1 score")
plt.xscale('log')
plt.show()