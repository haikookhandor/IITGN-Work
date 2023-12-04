# -*- coding: utf-8 -*-
"""Q3_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dWm-EWFDEwI15WbGIT0_bUIrxolF7Z1G
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linear_regression import LinearRegression
from os import path
import matplotlib.animation as animation
import os

N=60
x = np.array([i*np.pi/180 for i in range(60,300,2)])
np.random.seed(10)
y = 3*x + 8 + np.random.normal(0,3,len(x)) 

X = x.reshape(-1,1)
one_mat = np.ones((X.shape[0],1)) 
X = np.hstack((one_mat,X)) 


y=pd.Series(y)
LR = LinearRegression(fit_intercept=True)
LR.fit_gradient_descent(pd.DataFrame(X), y, batch_size=40,gradient_type='jax',penalty_type='l2',num_iters=50,lr=0.01)

a1= LR.all_coef[0]  
a2= LR.all_coef[1] 
for i in range(0,10):
    LR.plot_surface(pd.DataFrame(X),y,a1[i],a2[i]) 
    LR.plot_line_fit(pd.DataFrame(x),y,a1[i],a2[i]) 
    LR.plot_contour(pd.DataFrame(X),y,a1[i],a2[i]) 
