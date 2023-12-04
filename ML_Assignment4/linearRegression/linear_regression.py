# -*- coding: utf-8 -*-
"""linear_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dSwJGtzoxAFZlW8itoyHW0NtJSze1Pk7
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax
jax.config.update('jax_platform_name', 'cpu')
from sklearn.linear_model import LinearRegression as LR
import jax.numpy as jnp
from jax import grad
from sklearn.linear_model import SGDRegressor
np.random.seed(45)

import warnings
warnings.filterwarnings('ignore')

class LinearRegression():
  def __init__(self, fit_intercept = True):
    # Initialize relevant variables
    '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    '''
    self.fit_intercept = fit_intercept 
    self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
    self.all_coef=pd.DataFrame([]) # Stores the thetas for every iteration (theta vectors appended) (for the iterative methods)
    pass
  

  def fit_sklearn_LR(self, X, y):
    # Solve the linear regression problem by calling Linear Regression
    # from sklearn, with the relevant parameters
    lr = LR(fit_intercept=self.fit_intercept)
    lr.fit(X, y)
    self.coef_ = lr.coef_
    # pass
  
  def fit_normal_equations(self, X, y):
    # Solve the linear regression problem using the closed form solution
    # to the normal equation for minimizing ||Wx - y||_2^2
    self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
    # pass

  def fit_SVD(self, X, y):
      # Solve the linear regression problem using the SVD of the 
      # coefficient matrix
      U, s, V = np.linalg.svd(X, full_matrices=False)
      # U_pad = np.pad(U, ((0,0), (0, X.shape[0] - U.shape[0])), mode = 'constant')
      # VT_pad = np.pad(V.T, ((0, X.shape[1] - V.T.shape[0]), (0,0)), mode = 'constant')
      X_pinv = V@ np.diag(1/s) @ U.T
      self.coef_ = X_pinv @ y
      # pass

  def mse_loss(self, X, y):                
    # Compute the MSE loss with the learned model
    y_pred = self.predict(X)
    return np.mean((y_pred - y)**2)
    # pass

  def compute_gradient(self, X, y, penalty='unregularized', alpha=1):
      # Compute the analytical gradient (in vectorized form) of the 
      # 1. unregularized mse_loss,  and 
      # 2. mse_loss with ridge regularization
      # penalty :  specifies the regularization used  , 'l2' or unregularized
      if penalty == 'unregularized':
          grad = 2 * (X.T @ (X @ self.coef_ - y)) / X.shape[0]
      elif penalty == 'l2':
          regularization_term = alpha * self.coef_
          grad = 2 * (X.T @ (X @ self.coef_ - y)) / X.shape[0] + 2*regularization_term
      else:
          raise ValueError('penalty must be either unregularized or l2')
      return grad

  def compute_jax_gradient(self, X, y, penalty='unregularized', alpha = 1.0):
    # Compute the gradient of the 
    # 1. unregularized mse_loss, 
    # 2. mse_loss with LASSO regularization and 
    # 3. mse_loss with ridge regularization, using JAX 
    # penalty :  specifies the regularization used , 'l1' , 'l2' or unregularized
    X = jnp.array(X)
    y = jnp.array(y)
    if penalty == 'unregularized':
      mse_loss = lambda coef: jnp.mean((X @ coef - y)**2)
      grad_val = grad(mse_loss)(self.coef_)
    elif penalty == 'l1':
      mse_loss = lambda coef: jnp.mean((X @ coef - y)**2) + alpha * jnp.sum(jnp.abs(coef))
      grad_val = grad(mse_loss)(self.coef_)
    elif penalty == 'l2':
      mse_loss = lambda coef: jnp.mean((X @ coef - y)**2) + alpha * jnp.sum(coef**2)
      grad_val = grad(mse_loss)(self.coef_)
    else:
      raise ValueError('penalty must be either unregularized, l1 or l2')
    return grad_val

  def fit_gradient_descent(self, X, y, batch_size, alpha=0.1, gradient_type='manual', penalty_type='unregularized', num_iters=50, lr=0.01):
      # Implement batch gradient descent for linear regression (should unregularized as well as 'l1' and 'l2' regularized objective)
      # batch_size : Number of training points in each batch
      # num_iters : Number of iterations of gradient descent
      # lr : Default learning rate
      # gradient_type : manual or JAX gradients
      # penalty_type : 'l1', 'l2' or unregularized
      self.coef_ = np.zeros(X.shape[1])
      n_batches = int(np.ceil(X.shape[0] / batch_size))
      X = np.array(X)
      y = np.array(y)
      for i in range(num_iters):
        # shuffle the data
        idx = np.random.permutation(X.shape[0])
        for batch in range(0, X.shape[0], batch_size):
          # take a batch of data
          batch_idx = idx[batch:batch+batch_size]
          X_batch, y_batch = X[batch_idx], y[batch_idx]
          # print("1")
          # compute the gradient
          if gradient_type == "manual":
            # print("2")
            grad = self.compute_gradient(X_batch, y_batch, penalty_type, alpha)
            self.coef_ = self.coef_ - lr * grad
          elif gradient_type == 'jax':
          # else:
            # print("3")
            grad = self.compute_jax_gradient(X_batch, y_batch, penalty_type, alpha)
            self.coef_ = self.coef_ - lr * grad
            # print(self.coef_)
          # update the coefficients
        self.all_coef = self.all_coef.append(pd.Series(self.coef_), ignore_index=True)



  def fit_SGD_with_momentum(self, X,y, penalty='l2', alpha=0, lr=0.03, max_iter=1000, tol=1e-7, momentum=0.9):
      # Solve the linear regression problem using SGD with momentum
      # penalty: refers to the type of regularization used (ridge)
      
      self.coef_ = np.random.randn(X.shape[1])
      prev_mse = np.inf
      velocity = np.zeros_like(self.coef_)
      
      X = np.array(X)
      y = np.array(y)
      for epoch in range(max_iter):
          indices = np.random.permutation(X.shape[0])
          X = X[indices]
          y = y[indices]
          
          for i in range(X.shape[0]):
              xi = X[i]
              yi = y[i]
              # Compute the gradient of the loss function
              if penalty == 'l2':
                  grad_loss = 2 * (np.dot(self.coef_, xi) - yi) * xi + 2 * alpha * self.coef_
                  velocity = momentum * velocity + (1 - momentum) * grad_loss
                  self.coef_ -= lr * velocity

              elif penalty == 'unregularized':
                  grad_loss = 2 * (np.dot(self.coef_, xi) - yi) * xi
                  velocity = momentum * velocity + (1 - momentum) * grad_loss
                  self.coef_ -= lr * velocity
            
              
          # Compute the mean squared error and check for early stopping
          mse = np.mean((np.dot(X, self.coef_) - y) ** 2)
          if prev_mse - mse < tol:
              break
          prev_mse = mse
      pass


  def predict(self, X):
    # Funtion to run the LinearRegression on a test data point
    return X @ self.coef_ 
    pass


  def plot_surface(self, X, y, theta_0, theta_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param theta_0: Value of theta_0 for which to indicate RSS
        :param theta_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        theta0 = self.all_coef[0]
        theta1 = self.all_coef[1]

        # print(theta0,theta1)
        # theta0 = np.array(theta0)
        # theta1 = np.array(theta1)
        t0_min, t0_max, t1_min, t1_max = min(theta0), max(theta0), min(theta1), max(theta1)
        a = np.arange(t0_min-0.5, t0_max+1.5, (t0_max - t0_min)/50)
        b = np.arange(t1_min-0.5, t1_max+1.5, (t1_max - t1_min)/50)

        t0, t1= np.meshgrid(a,b)
        error_mat=[]
        for i, j in zip(t0,t1):
            error_mat.append(np.sum((y.values.reshape(len(y),1) - np.dot(X,pd.DataFrame([i,j])))**2, axis=0))
        error_mat = np.array(error_mat)
        
        y_predn=0
        for j in range(len(X)):
            y_predn+= (theta_0+(theta_1)*(X.iloc[j][0]) - y[j])**2
        errors= ((np.sum(y_predn)/len(X))**(1/2))

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(t0, t1, error_mat,cmap='plasma', edgecolor='none', alpha=0.5)
        ax.scatter(theta_0, theta_1, errors, color='r')
        plt.xlabel("theta 0")
        plt.ylabel("theta 1")
        ax.set_zlabel("Error")
        plt.title("Error = "+str(errors))
       # plt.savefig("plot_surface"+str(i))
        plt.show()
        pass

  def plot_line_fit(self, X, y, theta_0, theta_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of theta_0, theta_1. Plot must
        indicate theta_0 and theta_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param theta_0: Value of theta_0 for which to plot the fit
        :param theta_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        line  = theta_0 + theta_1*X
       
        plt.figure()
        plt.scatter(X,y, color='blue')  
        plt.plot(X, line,color='g')
       
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("theta0 = "+str(theta_0)+ " theta1 = "+str(theta_1))
       # plt.savefig("plt_line_fit"+str(i))
        plt.show()

        pass


  def plot_contour(self, X, y, theta_0, theta_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of theta_0 and theta_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param theta_0: Value of theta_0 for which to plot the fit
        :param theta_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        theta0= self.all_coef[0]
        theta1 = self.all_coef[1]
        # print(theta0,theta1)
        theta0_min,theta0_max = min(theta0),max(theta0) 
        theta1_min,theta1_max = min(theta1),max(theta1)

        a = np.arange(theta0_min-0.5, theta0_max+0.5, (theta0_max - theta0_min)/50)
        b = np.arange(theta1_min-0.5, theta1_max+0.5, (theta1_max - theta1_min)/50)

        t0, t1 = np.meshgrid(a,b)
        error_mat = []
        for i,j in zip(t0,t1):
            error_mat.append(np.sum((y.values.reshape(len(y),1) - np.dot(X,pd.DataFrame([i,j])))**2, axis=0))
        error_mat = np.array(error_mat)
        
        predict_y = 0
        for j in range(len(X)):
            predict_y += (theta_0 + (theta_1)*(X.iloc[j][0]) - y[j])**2
        err=((np.sum(predict_y)/len(X))**(1/2))
        
        plt.figure()
        ax = plt.axes()
        ax.contour(t0, t1, error_mat)
        ax.scatter(theta_0, theta_1, color='b')
        plt.xlabel("thetha 0")
        plt.ylabel("thetha 1")
        plt.title("Error = "+str(err))
        plt.show()

        pass