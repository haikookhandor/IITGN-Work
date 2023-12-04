import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
from sklearn.tree import DecisionTreeRegressor

# Data Preprocessing

df= pd.read_csv("auto-mpg.csv")
df.drop("car name",axis=1,inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'],errors = 'coerce')    # convert Horsepower from object to Float
df["horsepower"]= df["horsepower"].fillna(df["horsepower"].mean())    # fill missing values with mean
df = df.astype("float")                                                 # convert all fields to float
data_train = df.sample(frac=0.7, random_state=42)                         # Keeping 70% of the data for training
data_test = df.drop(data_train.index)                                     # Keeping 30% of the data for testing

print(data_train,data_test)

# Feature and Target Splitting

X_train= data_train.iloc[:,1:]                                      ## separating the features from the target
X_train= pd.DataFrame(X_train.values)                               ## converting the features to a dataframe

y_train=data_train.iloc[:,0].values                                 ## separating the target (mpg) from the features
y_train= pd.Series(y_train)

X_test= data_test.iloc[:,1:]                                        ## separating the features from the target
X_test = pd.DataFrame(X_test.values)

y_test=data_test.iloc[:,0].values                                   ## separating the target (mpg) from the features
y_test = pd.Series(y_test)


print(y_train)

# Decision Tree regressor using our implementation

dt_implement= DecisionTree(criterion='information_gain',max_depth=3)
dt_implement.fit(X_train,y_train)
y_implement = dt_implement.predict(X_test)

print("RMSE obtained from our implementation of Decision Tree is: ", rmse(y_implement,y_test))
print("MAE obtained from our implementation of Decision Tree is: ", mae(y_implement,y_test))


# Decision Tree regressor using Sklearn

dt_skl = DecisionTreeRegressor(max_depth=3)
dt_skl.fit(X_train, y_train)
y_skl = dt_skl.predict(X_test)


print("RMSE obtained from sklearn implementation is: ", rmse(y_skl,y_test))
print("MAE obtained from sklearn implementation is: ", mae(y_skl,y_test))