import numpy as np
def mae(y_hat, y):
    """
    Computes the mean absolute error (MAE) between predicted values (y_hat) and true values (y).
    :param y_hat: np.array of predicted values
    :param y: np.array of true values
    :return: float, mean absolute error
    """
    return np.mean(np.abs(y_hat - y))
def rmse(y_hat, y):
    """
    Computes the root mean square error (RMSE) between predicted values (y_hat) and true values (y).
    :param y_hat: np.array of predicted values
    :param y: np.array of true values
    :return: float, root mean square error
    """
    return np.sqrt(np.mean((y_hat - y)**2))