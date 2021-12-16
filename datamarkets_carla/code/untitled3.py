# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:45:26 2021

@author: shiuli Subhra Ghosh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
## Incorporating online SGD


class Online_QuantileSGD:
    import warnings
    warnings.filterwarnings("ignore")


    def __init__(self,  weight, bias, tow , learning_rate=0.2,damp_factor=1.02):
        self.learning_rate = learning_rate
        self.damp_factor = damp_factor
        self.w = weight
        self.b = bias
        self.tow = tow

    def fit_online(self, X, Y):

        # self.w = np.loadtxt(self.w_file).reshape(1, X.shape[1])  # Reading weights and bias
        # self.b = np.loadtxt(self.b_file).reshape(1, 1)


        for i in range(X.shape[0]):
            x = (np.array(X))[i].reshape(1, X.shape[1])
            y = (np.array(Y))[i].reshape(1, 1)
            Lw = np.dot((y - np.dot(x, self.w.T) - self.b), x)
            Lb = (y - np.dot(x, self.w.T) - self.b)
            self.w = self.w + self.learning_rate * Lw
            self.b = self.b + self.learning_rate * Lb
            self.learning_rate = self.learning_rate / self.damp_factor


        return self.w, self.b

    def fit_regression(self, data):
        formula=str(data.columns[-1])+' ~ '+" + ".join(data.columns[:-1])
        mod = smf.quantreg(formula,data)
        res = mod.fit(q=self.tow)
        self.b=res.params["Intercept"]
        self.w=np.array(list(res.params)[1:]).reshape(-1,1)

        return self.w, self.b

    def predict(self, X):
        # self.w = np.loadtxt(self.w_file).reshape(1, X.shape[1])  # Reading weights and bias
        # self.b = np.loadtxt(self.b_file).reshape(1, 1)
        m = np.dot(X, self.w.T) + self.b
        n = m.reshape(-1, )

        return n
