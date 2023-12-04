# -*- coding: utf-8 -*-
"""Q4_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dUjXq9nEOX2PMOboEnFAB_-x-hQGM49x
"""

import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linear_regression import LinearRegression
import pandas as pd
np.random.seed(45)  #Setting seed for reproducibility

# if not path.exists('Plots/Question4/'):
#   os.makedirs('Plots/Question4/')

x = np.array([i*np.pi/180 for i in range(60,300,2)])
y = 3*x + 8 + np.random.normal(0,3,len(x))

#TODO : Write here
#Preprocess the input using the polynomial features
#Solve the resulting linear regression problem by calling any one of the 
#algorithms you have implemented.


def LR_normal(X, y):
    XT = X.T
    X = np.array(X)
    XT = np.array(XT)
    y_1 = np.array(y)
    theta = ((np.linalg.inv(XT.dot(X))).dot(XT)).dot(y_1) # Normal Equation
    return theta

degrees=[]
for i in range(10):
    degrees.append(i)

#varying degrees
theta_norm = []
for d in degrees:
    X = []
    for i in range(len(x)):
        poly = PolynomialFeatures(d)
        X.append(poly.transform([x[i]]))

    X = pd.DataFrame(X)
    y = pd.Series(y)
    theta = LR_normal(X,y)
    theta_norm.append(np.linalg.norm(theta))

#Plotting
plt.figure()
plt.plot(degrees,theta_norm)
plt.title("Theta v/s degrees")
plt.xlabel("degrees")
plt.ylabel("|theta|")
plt.show()