import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd


class Online_SGD:
    def __init__(self, X, y, learning_rate=0.2):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.w = np.zeros((1, self.X.shape[1]))  # Randomly initializing weights
        self.b = np.zeros((1, 1))

    def fit(self):
        for i in range(self.X.shape[0]):
            Lw = (self.X[i]) * (self.y[i] - np.dot(self.X[i], self.w.T) - self.b)
            Lb = (self.y[i] - np.dot(self.X[i], self.w.T) - self.b)
            self.w = self.w - self.learning_rate * Lw
            self.b = self.b - self.learning_rate * Lb
            self.learning_rate = self.learning_rate / 1.02

        return self.w, self.b

    def transform(self):
        return np.dot(self.X, self.w.T) + self.b

    def error(self):
        y_pred = np.dot(self.X, self.w.T) + self.b
        loss = mean_squared_error(y_pred, self.y)
        print(loss)

    def plot(self):
        from matplotlib.pyplot import figure
        y_pred = np.dot(self.X, self.w.T) + self.b
        plt.figure(figsize=(25, 6))
        plt.plot(self.y, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend(prop={'size': 16})
        plt.show()
        print('Mean Squared Error :', mean_squared_error(y_pred, self.y))



# data = pd.read_csv("dataset.csv")
# y = data["Profit"]
# X = np.array(data["Population"])
# X1 = X.reshape(X.shape[0], -1)
# lr = Online_SGD(X1, y)
# lr.fit()
# lr.transform()
