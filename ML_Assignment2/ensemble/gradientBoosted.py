from sklearn.tree import DecisionTreeRegressor
import numpy as np
class GradientBoostedRegressor():
    def __init__(self, base_estimator=DecisionTreeRegressor, n_estimators=3, learning_rate=0.1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.coef = []

    def fit(self, X, y):
        f = y.mean()  # Initial prediction is the mean of y
        
        for m in range(self.n_estimators):
            residual = y - f
            model = self.base_estimator()
            model.fit(X, residual)
            self.models.append(model)
            self.coef.append(self.learning_rate)
            f += self.learning_rate * model.predict(X)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, model in enumerate(self.models):
            y_pred += self.coef[i] * model.predict(X)
        return y_pred