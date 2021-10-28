import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd


class Online_SGD:

    def __init__(self, learning_rate=0.2, n_epochs=100, k=40, damp_factor=1.02):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.k = k
        self.damp_factor = damp_factor

    def fit(self, X, Y):
        self.w = np.zeros((1, X.shape[1]))  # Randomly initializing weights
        self.b = np.zeros((1, 1))
        #         print("Shape of W",self.w.shape)
        #         print("Shape of b",self.b.shape)
        #         print("W",self.w)
        #         print("b",self.b)
        for i in range(X.shape[0]):
            #             print("shape of X[i]",X[i].shape)
            #             print("shape of Y[i]",Y[i].shape)
            #             print("shape of w.T",self.w.T.shape)
            x = (np.array(X))[i].reshape(1, X.shape[1])
            y = (np.array(Y))[i].reshape(1, 1)
            #             print("shape of x",x.shape)
            #             print("shape of y]",y.shape)
            Lw = np.dot((y - np.dot(x, self.w.T) - self.b), x)
            Lb = (y - np.dot(x, self.w.T) - self.b)
            self.w = self.w + self.learning_rate * Lw
            self.b = self.b + self.learning_rate * Lb
            #             print("W",self.w)
            #             print("b",self.b)
            self.learning_rate = self.learning_rate / self.damp_factor
        return self.w, self.b

    def fit_batch(self, X, Y):
        self.w = np.zeros((1, X.shape[1]))  # Randomly initializing weights
        self.b = np.zeros((1, 1))
        n = 1
        while n <= self.n_epochs:
            temp = pd.merge(pd.DataFrame(X), pd.DataFrame(Y), how='left')
            temp2 = temp.sample(self.k)
            X_tr = temp2.iloc[:, 0:temp.shape[1]].values
            Y_tr = temp2.iloc[:, -1].values

            for i in range(self.k):
                x = X_tr[i].reshape(1, X_tr.shape[1])
                y = Y_tr[i].reshape(1, 1)
                #                 print("shape of x",x.shape)
                #                 print("shape of y",y.shape)
                Lw = (np.dot((y - np.dot(x, self.w.T) - self.b), x)) * (2 / self.k)
                Lb = ((y - np.dot(x, self.w.T) - self.b)) * (2 / self.k)
                self.w = self.w + self.learning_rate * Lw
                self.b = self.b + self.learning_rate * Lb
            #             print("W",self.w)
            #             print("b",self.b)

            self.learning_rate = self.learning_rate / 1
            n = n + 1
        return self.w, self.b

    def predict(self, X):
        m = np.dot(X, self.w.T) + self.b
        n = m.reshape(-1, )
        return n