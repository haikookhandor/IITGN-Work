from typing import Union
import pandas as pd
import math

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    Inputs:
    1. y_hat: pd.Series of predictions
    2. y: pd.Series of ground truth
    
    Output: 
       Returns the accuracy as float
    """
    
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size) # Sanity check to ensure that y_hat and y are of same size
    
    if isinstance(y,pd.Series): #if y is a Pandas series, convert to numpy array
        y= y.values 
    corr_classify = sum(y_hat == y) # Number of correct classifications
    total_pred = len(y_hat) # Total number of predictions
    return float((corr_classify / total_pred))

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision

    Inputs:
    1. y_hat: pd.Series of predictions
    2. y: pd.Series of ground truth
    3. cls: The class chosen
    Output:
       Returns the precision as float
    """
    assert(y_hat.size == y.size) # Sanity check to ensure that y_hat and y are of same size
    
    if isinstance(y,pd.Series): #if y is a Pandas series, convert to numpy array
        y= y.values
    true_pos=0.0
    false_pos=0.0
    for i in range(len(y)):
        if(y_hat[i]==cls): # if the predicted class is same as the class chosen
            if(y[i]==cls): # if the ground truth is also same as the class chosen
                true_pos+=1
            elif(y[i]!=cls): # if the ground truth is not the same as the class chosen
                false_pos+=1
    if float(true_pos)==0:
        return 0.0
    return (true_pos/float(true_pos+false_pos))
  

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall

    Inputs:
    1. y_hat: pd.Series of predictions
    2. y: pd.Series of ground truth
    3. cls: The class chosen
    Output:
       Returns the recall as float
    """
    assert(y_hat.size == y.size) # Sanity check to ensure that y_hat and y are of same size
    
    if isinstance(y,pd.Series): #if y is a Pandas series, convert to numpy array
        y= y.values
        
    true_pos=0.0
    false_neg=0.0
    for i in range(len(y_hat)):
        if(y[i]==cls): # if the ground truth is same as the class chosen
            if(y_hat[i]==cls): # if the predicted class is also same as the class chosen
                true_pos+=1
                
            elif(y_hat[i]!=cls): # if the predicted class is not the same as the class chosen
                false_neg+=1
    if float(true_pos)==0:
        return 0.0                
    return (true_pos/float(true_pos+false_neg))

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    1. y_hat: pd.Series of predictions
    2. y: pd.Series of ground truth
    Output:
      Returns the rmse as float
    """
    assert(y_hat.size==y.size) # Sanity check to ensure that y_hat and y are of same size
    
    if isinstance(y, pd.Series): #if y is a Pandas series, convert to numpy array
        y = y.values
    
    square_err = (y-y_hat)**2
    mean_square_err = float(sum(square_err))/len(y)
    return (math.sqrt(mean_square_err))
    

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    1. y_hat: pd.Series of predictions
    2. y: pd.Series of ground truth
    Output:
       Returns the mae as float
    """
    assert(y_hat.size==y.size) # Sanity check to ensure that y_hat and y are of same size
    
    if isinstance(y, pd.Series): #if y is a Pandas series, convert to numpy array
        y = y.values
    absolute_err = abs(y-y_hat)
    return (float(sum(absolute_err))/len(y))
