import pandas as pd
import numpy as np
import math


def entropy(Y: pd.Series, weights: pd.Series = None) -> float:
    """
    Function to calculate the entropy of a Pandas Series with weighted samples
    """
    y_list = list(Y)
    y_size = len(y_list)

    if weights is not None: # If weights are provided           
        weights = list(weights)
        weights_sum = sum(weights)     
        counts = {}                     
        for i, j in enumerate(y_list):
            if j not in counts:
                counts[j] = 0
            counts[j] += weights[i]     

        entropy = 0 # Entropy initialization
        for i in counts: 
            p = counts[i] / weights_sum  # weighted probability
            if p > 0:
                entropy -= p * math.log2(p)


    else:                     # if samples are not weighted
        count = {}          
        for i in y_list:
            if i in count:
                count[i] += 1          
            else:
                count[i] = 1

        entropy = 0 # Entropy initialization
        for i in count:
            p = count[i] / y_size        
            entropy -= p * math.log2(p)  

    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    count = {}
    y_list = list(Y)
    y_size = Y.size

    for i in y_list:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1 

    gini_ind = 1 # Maximum possible gini index
    for i in count:
        p = count[i] / y_size
        gini_ind -= p**2

    return gini_ind 


def information_gain(Y: pd.Series, attr: pd.Series, weights=None) -> float:
    """
    Function to calculate the information gain
    """
    count = {}            
    y_list = list(Y)
    y_size = len(y_list)
    attr_list = list(attr)

    if weights is not None:           # if the samples are weighted 
        weights = list(weights)
        weights_sum = sum(weights)
        
        for i in range(attr.size):
            if attr_list[i] in count:
                count[attr_list[i]].append((y_list[i], weights[i]))
            else:
                count[attr_list[i]] = [(y_list[i], weights[i])]        
                
        info_gain = entropy(y_list, weights)       
        for i in count:
            i_counts = [j[0] for j in count[i]]             
            i_weights = [j[1] for j in count[i]]
            info_gain -= (sum(i_weights) / weights_sum) * entropy(i_counts, i_weights)          
            
    else:                          # if the samples are not weighted
        for i in range(attr.size):             
            if attr_list[i] in count:
                count[attr_list[i]].append(y_list[i])  
            else:
                count[attr_list[i]] = [y_list[i]]

        info_gain = entropy(y_list)
        for i in count:
            info_gain -= (len(count[i]) / y_size) * entropy(count[i])      

    return info_gain

