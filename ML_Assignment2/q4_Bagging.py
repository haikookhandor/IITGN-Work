import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from metrics import *

from ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Or use sklearn decision tree

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")


warnings.filterwarnings("ignore")
criteria = "entropy"

tree = DecisionTreeClassifier
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, criterion = criteria ,n_jobs = 1)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot(X,y)
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print('Precision for', cls,': ', precision(y_hat, y, cls))
    print('Recall for',cls,':', recall(y_hat, y, cls))
