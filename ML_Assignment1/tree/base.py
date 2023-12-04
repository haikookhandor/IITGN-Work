"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class TreeNode():
    def __init__(self):
        self.param_val = None # Parameter value
        self.isLeaf = False # Check if node is leaf
        self.attr_number = None # Store feature number of split
        self.split_value = None # Store the split value
        self.isAttritbute= False # True for categorical data, otherwise False
        self.tree_child = {} # Store children 

# Separate class for Regression and Classification
# Regression class
class DecisionTreeRegressor(): 
    
    def __init__(self, criteria, max_depth = None):
        self.critera = criteria # Criteria for split
        self.max_depth = max_depth # Maximum depth of tree
        self.head = None # Current head of tree
        
    def split_fit(self, X, y, depth):
        curr_Node = TreeNode()   # New Node
        curr_Node.attr_number = -1 # Variable to store feature number of split as -1 initially i.e. it is the root node and no other nodes are present
        splitval = None # Variable to store the split value 
        criteria_val = None # Variable to store the final criteria value      
          
        classes = np.unique(y) 
        # covering the base/edge cases like zero features or max_depth is specified or all the classes are same
        
        # if zero features exist
        if X.shape[1]==0:
            curr_Node.isLeaf = True # If no features exist, then it is a leaf node
            curr_Node.param_val = y.mean() # Store the mean of y as the parameter value 
            return curr_Node

        # if max_depth is specified
        if self.max_depth!=None:
            if depth==self.max_depth: 
                curr_Node.isLeaf = True # If depth is equal to max_depth, then it is a leaf node for us
                curr_Node.param_val = y.mean() # Store the mean of y as the parameter value
                return curr_Node
        
        # if all the classes are same
        if len(classes)==1: 
            curr_Node.isLeaf = True # If all the classes are same, then it is a leaf node
            curr_Node.param_val = classes[0] # Store the one class as the parameter value
            return curr_Node

        for feature in X:
            x = X[feature] # Store the feature in x
            # Discrete Input V/s Real Input
            if x.dtype.name=="category": # Discrete Input 
                unique_classes = np.unique(x)
                fin_value = 0
                
                for j in unique_classes: 
                    y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j]) # For each row in x that has class j, create a sublist of y.
                    fin_value += (y_sub.size)*np.var(y_sub) # weighted addition of variance of each class
                    
                if criteria_val==None:
                    criteria_val = fin_value # Store the final criteria value
                    attr_no = feature 
                    splitval = None # absent in case of categorical data
                else:
                    if criteria_val>fin_value:
                        criteria_val = fin_value
                        attr_no = feature
                        splitval = None
            
            else:  # Real Input 
                x_sorted = x.sort_values() # Sorting the values of x
                
                for j in range(len(y)-1): 
                    index = x_sorted.index[j] # finding the best split by mean of consecutive values
                    next_index = x_sorted.index[j+1]
                    split_value = (x[index]+x[next_index])/2 # mean of index and index + 1
                    y_sub1 = pd.Series([y[k] for k in range(y.size) if x[k]<=split_value]) # For each row in x that has value less than split_value, create a sublist of y.
                    y_sub2 = pd.Series([y[k] for k in range(y.size) if x[k]>split_value]) # For each row in x that has value greater than split_value, create a sublist of y.
                    fin_value = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2) # weighted addition of variance of each class
                    
                    if criteria_val==None :
                        attr_no = feature 
                        criteria_val = fin_value
                        splitval = split_value
                    else:
                        if fin_value<criteria_val:
                            attr_no = feature
                            criteria_val = fin_value
                            splitval = split_value
    

    # Current attribute can be categorical or Real
    
        if splitval==None: # if current node feature is categorical
            
            curr_Node.attr_number = attr_no 
            curr_Node.isAttritbute = True
            classes = np.unique(X[attr_no]) # Find the number of classes in the current attribute
            
            for j in classes: 
                y_new = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype) # For each row in x that has class j, create a sublist of y.
                X_new = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1) 
                curr_Node.tree_child[j] = self.split_fit(X_new, y_new, depth+1) # Recursion step
                
        else: # if current node feature is real 
            curr_Node.attr_number = attr_no
            curr_Node.split_value = splitval
            y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=splitval], dtype=y.dtype)
            X_new1 = X[X[attr_no]<=splitval].reset_index().drop(['index'],axis=1)
            y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>splitval], dtype=y.dtype)
            X_new2 = X[X[attr_no]>splitval].reset_index().drop(['index'],axis=1)
            curr_Node.tree_child["lessThan"] = self.split_fit(X_new1, y_new1, depth+1)
            curr_Node.tree_child["greaterThan"] = self.split_fit(X_new2, y_new2, depth+1)
        return curr_Node


    def fit(self, X, y): # Fit function
        if X.shape[0]==len(y) & len(y)>0: 
            self.head = self.split_fit(X,y,0) 
        return self.head 

    def predict(self, X): # Predict function
        y_hat = []                  

        for i in range(X.shape[0]): 
            x_row = X.iloc[i,:] # For each row in X   
            node = self.head 
            while not node.isLeaf:                            
                if node.isAttritbute:                               
                    node = node.tree_child[x_row[node.attr_number]]
                else:                                       
                    if x_row[node.attr_number]>node.split_value:
                        node = node.tree_child["greaterThan"]
                    else:
                        node = node.tree_child["lessThan"]
            
            y_hat.append(node.param_val)                           
        
        y_hat = pd.Series(y_hat)
        return y_hat
    
    def plotTree(self, root, depth): # Plot function for decision tree works recursively by traversing the tree
            if root.isLeaf: # If the node is a leaf node
                if root.isAttritbute:
                    return "Class "+str(root.param_val)
                else:
                    return "Value "+str(root.param_val)

            a = ""
            if root.isAttritbute:
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number)+" <= "+str(root.split_value)+")\n"
                a += "\t"*(depth+1)
                a += "Y: " + str(self.plotTree(root.tree_child["lessThan"], depth+1)).rstrip("\n") + "\n"
                a += "\t"*(depth+1)
                a += "N: " + str(self.plotTree(root.tree_child["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
            return a


# Classification Class
class DecisionTreeClassifier(): 
    
    def __init__(self, criteria, max_depth=None):
        self.criteria = criteria 
        self.max_depth = max_depth
        self.head = None
        
    def split_fit(self, X, y, depth):
        curr_Node = TreeNode()   # New Node
        curr_Node.attr_number = -1 # Variable to store feature number of split as -1 initially i.e. it is the root node and no other nodes are present 
        splitval = None # Store split value
        criteria_val = None # Store final criteria value           
        classes=np.unique(y)

        # covering the base/edge cases like zero features or max_depth is specified or all the classes are same
        
        #if zero features exist
        if X.shape[1]==0: 
                curr_Node.isLeaf = True
                curr_Node.isAttritbute = True
                curr_Node.param_val = y.value_counts().idxmax() # Mode of all outputs, maximal voting/counting 
                return curr_Node

        # if a single value of y exists
        if len(classes)==1: 
                curr_Node.isLeaf = True 
                curr_Node.isAttritbute = True 
                curr_Node.param_val = classes[0] 
                return curr_Node

        #if max depth is specified
        if self.max_depth!=None: 
                if self.max_depth==depth: 
                    curr_Node.isLeaf = True 
                    curr_Node.isAttritbute = True
                    curr_Node.param_val = y.value_counts().idxmax() # Output is determined by maximal voting/counting
                    return curr_Node
           
        for feature in X: 
                x = X[feature] 
                # Discrete input or Real input
                # Discrete input
                if x.dtype.name=="category": 
                    fin_value = None
                    if self.criteria=="gini_index": #Calculating the gini index    
                        classes_gini = np.unique(x)
                        s1 = 0
                        for j in classes_gini:
                            y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j]) # Subsetting the y values based on the feature value
                            s1 += (y_sub.size)*gini_index(y_sub) 
                        fin_value = -1*(s1/x.size) 
        
                    else: #Calculating the Information Gain
                        fin_value = information_gain(y,x)
                        
                    if criteria_val==None: 
                            attr_no = feature
                            criteria_val = fin_value
                            splitval = None
                    else:
                       
                        if criteria_val<fin_value:
                            attr_no = feature
                            criteria_val = fin_value
                            splitval = None
                
                # For Real Input 
                else:
                    x_sorted = x.sort_values() # Sorting the values of the feature
                    for j in range(len(x_sorted)-1):
                        index = x_sorted.index[j]
                        next_index = x_sorted.index[j+1]

                        if y[index]!=y[next_index]:
                            fin_value = None
                            split_value = (x[index]+x[next_index])/2 # find the split by taking a average of index and index + 1
                            
                            if self.criteria=="information_gain": 
                                info_attr = pd.Series(x<=split_value)
                                fin_value = information_gain(y,info_attr)
                                
                            else:                                              
                                y_sub1 = pd.Series([y[k] for k in range(len(y)) if x[k]<=split_value])
                                y_sub2 = pd.Series([y[k] for k in range(len(y)) if x[k]>split_value])
                                fin_value = y_sub1.size*gini_index(y_sub1) + y_sub2.size*gini_index(y_sub2)
                                fin_value =  -1*(fin_value/y.size)
                                
                            if criteria_val==None:
                                attr_no = feature
                                criteria_val = fin_value
                                splitval = split_value
                            else:
                                if criteria_val<fin_value:
                                    attr_no = feature
                                    criteria_val = fin_value
                                    splitval = split_value
                                    
                if splitval==None:
                
                    curr_Node.attr_number = attr_no
                    curr_Node.isAttritbute = True
                    classes = np.unique(X[attr_no])
                    
                    for j in classes:
                        y_new = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype)
                        X_new = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1)
                        curr_Node.tree_child[j] = self.split_fit(X_new, y_new, depth+1) # Recursive call to split_fit function
                
                else:
                    curr_Node.attr_number = attr_no
                    curr_Node.split_value = splitval
                                   
                    y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=splitval], dtype=y.dtype)
                    X_new1 = X[X[attr_no]<=splitval].reset_index().drop(['index'],axis=1)
                    y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>splitval], dtype=y.dtype)
                    X_new2 = X[X[attr_no]>splitval].reset_index().drop(['index'],axis=1)
                    curr_Node.tree_child["lessThan"] = self.split_fit(X_new1, y_new1, depth+1)
                    curr_Node.tree_child["greaterThan"] = self.split_fit(X_new2, y_new2, depth+1)
        return curr_Node    
        

    def fit(self, X, y): # Fit function
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        self.head = self.split_fit(X,y,0)
        return self.head
        
    def predict(self, X): # Predict function
        y_hat = []                  

        for i in range(X.shape[0]):
            x_row = X.iloc[i,:]    # For every row of X
            node = self.head
            while not node.isLeaf:                         
                if node.isAttritbute:      
                    node = node.tree_child[x_row[node.attr_number]]
                else:                                   
                    if x_row[node.attr_number]>node.split_value:
                        node = node.tree_child["greaterThan"]
                    else:
                        node = node.tree_child["lessThan"]
            
            y_hat.append(node.param_val)                           
        
        y_hat = pd.Series(y_hat)
        return y_hat
    
    def plotTree(self, root, depth): # Plot function for decision tree works recursively by traversing the tree
            if root.isLeaf: 
                if root.isAttritbute:
                    return "Class "+str(root.param_val)
                else:
                    return "Value "+str(root.param_val)

            a = ""
            if root.isAttritbute:
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number)+" <= "+str(root.split_value)+")\n"
                a += "\t"*(depth+1)
                a += "Y: " + str(self.plotTree(root.tree_child["lessThan"], depth+1)).rstrip("\n") + "\n"
                a += "\t"*(depth+1)
                a += "N: " + str(self.plotTree(root.tree_child["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
            return a
        


class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # this criterion is not for regression
        > max_depth : The maximum depth of the tree
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None
        self.tree = None

    def fit(self, X, y): # Fit function
        """
        Function for training and constructing the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features. Shape of X is (N x P) where N is the number of samples and P is the number of columns
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if y.dtype.name=="category":
            self.tree =DecisionTreeClassifier(criteria=self.criterion, max_depth=self.max_depth)
            self.head = self.tree.fit(X, y)
        else:
            self.tree =DecisionTreeRegressor(criteria=self.criterion, max_depth=self.max_depth) 
            self.head = self.tree.fit(X, y)


    def predict(self, X): # Predict function
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = self.tree.predict(X)                 
        return y_hat
        

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No

        """
        plot1= self.tree.plotTree(self.head,depth=0)
        print(plot1)