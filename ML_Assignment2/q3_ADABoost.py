import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from ensemble.ADABoost import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 300
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = "entropy"
AdaClassifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criteria, max_depth=1), n_estimators=n_estimators)
AdaClassifier.fit(X, y)

y_hat = AdaClassifier.predict(X)

[fig1, fig2] = AdaClassifier.plot(X,y)

print("Accuracy = ", accuracy(y_hat, y))
print("Criteria = ", criteria)

for cls in y.unique():
    print('Precision for ',cls,"=", precision(y_hat, y, cls))
    print('Recall for ', cls, "=", recall(y_hat, y, cls))



##### AdaBoostClassifier on Classification data set using the entire data set

from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000,
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)


X=pd.DataFrame(X)
X['y']=y

data_f= X.sample(frac=1, random_state=42)
data_f.reset_index(drop=True, inplace=True)
y= data_f.pop('y')

split_range= int(0.7*len(X))
X_train= data_f[:split_range]
X_test = data_f[split_range:]
y_train= pd.Series(y[:split_range],dtype=y.dtype)
y_test = pd.Series(y[split_range:],dtype=y.dtype)

n_estimators=3
tree= DecisionTreeClassifier(criterion='entropy',max_depth=1)
clf = AdaBoostClassifier(base_estimator=tree, n_estimators= n_estimators)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf.plot(X,y)

print("Accuracy = ", accuracy(y_pred,y_test))

for uni in y.unique():
    print('Precision for',uni,' : ', precision(y_pred, y_test, uni))
    print('Recall for ',uni ,': ', recall(y_pred, y_test, uni))