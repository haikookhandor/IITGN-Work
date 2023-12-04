# Bias-Variance-Tradeoff

Bias: The difference between the average of our predictions and the correct values which we are trying to predict.

Variance: It is the variance or the expectation of the squared distance between our predictions and the mean prediction. 

## Decision Trees

Max_depth: int type of variable, specifies the maximum depth of the decision tree. If not specified, it builds the tree till only pure classes are left as leaf nodes.

If Max_depth is <<< i.e. low, the model learnt cannot capture the underlying structure of the data on which we fit the model. This results in the model being simple, over-generalizes on the test_data and has an inherent high bias. 

If Max_depth is >>> i.e. high, the model very intricately captures all details of the data on which it is fit including any outliers or noise which is not optimal. This results in the model being highly complex, highly specific i.e. not general at all on test_data. Even if the training data is changed slightly, the new model learnt is quite different i.e. it has a high variance.