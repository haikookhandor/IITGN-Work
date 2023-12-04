import pandas as pd
import numpy as np
import math

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy 

    Inputs:
      Y: pd.Series of Labels
    Outpus:
      Returns the entropy as a float
    """
    entropy = 0.0 # initialize entropy
    freq = np.bincount(Y) # Store count of unique value from the passed column
    probabilities = freq / len(Y) # probability of each class
    for p in probabilities:
        if p > 0.0: 
            entropy += p*math.log(p,2)
    return -entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index

    Inputs:
      Y: pd.Series of Labels
    Outpus:
      Returns the gini index as a float
    """
    gini = 0.0
    freq = np.bincount(Y) #returns a NumPy array that stores count of unique value from the passed column.
    probabilities = freq / len(Y) #find probability of each class
    for p in probabilities:
        gini+= p**2
    return 1-gini

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    
    Inputs:
    1. Y: pd.Series of Labels
    2. attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
      Return the information gain as a float
    """
    store = {} # dictionary to store the attribute values and their corresponding labels
    for i in range(len(attr)):
        if attr[i] in store:
            store[attr[i]].append(Y[i])
        else:
            store[attr[i]] = [Y[i]]
            
    info_gain = 0
    for i in store: # calculate the information gain
        info_gain += (len(store[i])/len(Y))*entropy(store[i])
    info_gain = entropy(Y) - info_gain
    return info_gain
