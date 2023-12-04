import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tree.base import DecisionTree


np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps


X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.2, random_state=1)
max_depths = range(1, 11)
n_trees = 100
biases = []
vars = []

for depth in tqdm(max_depths):
    model = DecisionTree(max_depth=depth)
    predictions = []
    for i in range(n_trees):
        idx = np.random.choice(np.arange(len(X_train)), size=len(X_train), replace=True)
        x_new, y_new = X_train[idx], y_train[idx]
        x_new = pd.DataFrame(x_new)
        y_new = pd.Series(y_new)
        model.fit(x_new, y_new)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    predictions_mean = np.mean(predictions, axis=0)
    biases.append(np.mean((y_test - predictions_mean)**2))
    vars.append(np.mean((predictions - np.mean(predictions))**2))

nor_var = (vars - np.min(vars))/(np.max(vars) - np.min(vars))
nor_bias = (biases - np.min(biases))/(np.max(biases) - np.min(biases))

plt.plot(max_depths, nor_bias, label='Bias')
plt.plot(max_depths, nor_var, label='Variance')
# plt.plot(max_depths, biases, label='Bias')
# plt.plot(max_depths, vars, label='Variance')
plt.legend()
plt.xlabel('Maximum Depth')
plt.ylabel('Error')
plt.savefig('figures/q1_bias_variance.png')