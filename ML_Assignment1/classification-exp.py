# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from tqdm import tqdm # For progress bar

np.random.seed(42)

# Read dataset
# ...
# 
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5) # Dataset
# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("plots/data_visualisation.png")

split_rng= int(0.7*len(X)) # 70% of data for training, 30% for testing

X_train= pd.DataFrame(X[:split_rng])
X_test = pd.DataFrame(X[split_rng:])
y_train= pd.Series(y[:split_rng],dtype=y.dtype)
y_test = pd.Series(y[split_rng:],dtype=y.dtype)


clf= DecisionTree(criterion='information_gain',max_depth=5) # Decision Tree
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Test Accuracy: ", accuracy(y_pred,y_test)) # Test Accuracy
for classes in list(set(y)):
    print(f"Precision, class {str(classes)} : {precision(y_pred,y_test,classes)})") # Precision
    print(f"Recall, class {str(classes)} : {recall(y_pred,y_test,classes)})")  # Recall

# b part
cross_val= KFold(n_splits=5, random_state=1, shuffle=True) # 5 fold cross validation
optimal_values= {0:dict(), 1:dict(), 2:dict(), 3:dict(), 4:dict()} # Dictionary to store optimum depth and accuracy for each fold

for Fold_No, (train_idx,test_idx) in tqdm(enumerate(cross_val.split(X=X,y=y))):
    X_train, X_val= pd.DataFrame(X[train_idx]), pd.DataFrame(X[test_idx])
    y_train,y_val = y[train_idx], y[test_idx]
    # Initialize max_accuracy and optimal depth
    max_accuracy = 0
    opt_d = 2
    
    for depth in range(2,10): # Iterate over depths
        clf = DecisionTree(criterion='gini-index',max_depth=depth)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy(y_pred,y_val)
        if(acc>max_accuracy): # Update max_accuracy and optimal depth
            opt_d = depth
            max_accuracy = acc
    optimal_values[Fold_No][opt_d]= max_accuracy # Store optimal depth and accuracy for each fold
    
    
best_acc=0
best_depth=0
for key, value in optimal_values.items():
    for key1, val in value.items():
        print("Fold:"+str(key)+" "+"Optimal Depth:"+str(key1)+ " "+"Accuracy:"+str(val)) # Printing to check optimal depth and accuracy for each fold
        
for key,value in optimal_values.items():
    for key1,val in value.items():
        if(val>best_acc): # Finding best depth and accuracy
            best_acc = val
            best_depth = key1
            
print("Best Depth is ", best_depth)
print("Best Accuracy is ", best_acc)

    
    