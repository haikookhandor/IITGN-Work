from typing import Union
import pandas as pd
import math

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    if isinstance(y,pd.Series): 
        y= y.values 
    correct_classify = sum(y_hat == y)         # Correctly classified number
    total_classify = len(y_hat)                 # Total Predictions    
    return float((correct_classify / total_classify))

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    if isinstance(y,pd.Series): 
        y= y.values
    true_positive=0.0
    false_negative=0.0
    for i in range(len(y_hat)):
        if(y[i]==cls): 
            if(y_hat[i]==cls): 
                true_positive+=1
                
            elif(y_hat[i]!=cls): # if the predicted class is not the same as the class chosen
                false_negative+=1
    if float(true_positive)==0:
        return 0.0                
    return (true_positive/float(true_positive+false_negative))

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    if isinstance(y,pd.Series): 
        y= y.values
    true_positive=0.0
    false_positive=0.0
    for i in range(len(y)):
        if(y_hat[i]==cls): 
            if(y[i]==cls): 
                true_positive+=1
            elif(y[i]!=cls): 
                false_positive+=1
    if float(true_positive)==0:
        return 0.0
    return (true_positive/float(true_positive+false_positive))
  
def mae(y_hat: pd.Series, y: pd.Series) -> float:
    if isinstance(y, pd.Series): 
        y = y.values
    absolute_error = abs(y-y_hat)
    return (float(sum(absolute_error))/len(y))


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    if isinstance(y, pd.Series): 
        y = y.values
    
    error_square = (y-y_hat)**2
    mean_error_square = float(sum(error_square))/len(y)
    return (math.sqrt(mean_error_square))