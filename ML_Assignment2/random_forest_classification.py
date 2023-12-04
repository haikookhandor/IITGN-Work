import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


###Write code here

X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

y_train_cat = y_train_series.astype('category')
y_test_cat = y_test_series.astype('category')  

X_df = pd.DataFrame(X)
y_series = pd.Series(y)
y_cat = y_series.astype('category')

# Learning classifier
tree = RandomForestClassifier(10, criterion="information_gain")
tree.fit(X_train_df, y_train_cat)

# Predicition
y_hat = tree.predict(X_test_df)
print("Please wait, preparing plots...", flush=True)
tree.plot(X_df, y_cat)
# tree.decisionBoundary(X_df,y_cat)

# print("ok")

# Calculating Metrics
print('Accuracy: ', accuracy(y_hat, y_test_cat))
for cls in y_series.unique():
    print('Precision: ', precision(y_hat, y_test_cat, cls))
    print('Recall: ', recall(y_hat, y_test_cat, cls))
