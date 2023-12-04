import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
import random

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn.datasets import make_classification

np.random.seed(42)
random.seed(42)

########### RandomForestClassifier ###################

N = 300
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

X, y = make_classification(n_samples=50,n_features=5, n_redundant=0, n_informative=4, random_state=15, n_clusters_per_class=1, class_sep=0.5, n_classes=5)
X = pd.DataFrame(X)
y = pd.Series(y)

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot(X,y)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########## RandomForestRegressor ###################


N = 300
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


Regressor_RF = RandomForestRegressor(10, criterion = 'variance')
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('Criteria :', 'variance')
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
