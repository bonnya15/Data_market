# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:45:26 2021

@author: shiuli Subhra Ghosh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
## Incorporating online SGD


class Online_QuantileSGD:
    import warnings
    warnings.filterwarnings("ignore")


    def __init__(self,  weight, bias, tow , alpha, learning_rate=0.2,damp_factor=1.02):
        self.learning_rate = learning_rate
        self.damp_factor = damp_factor
        self.w = weight
        self.b = bias
        self.tow = tow
        self.alpha = alpha

    def fit_online(self, data):
        Y = data.iloc[:,-1]
        X = data.iloc[:,0:-1]

        for i in range(X.shape[0]):
            x = (np.array(X))[i].reshape(1, X.shape[1])
            y = (np.array(Y))[i].reshape(1, 1)
            bias = self.b.reshape(1,1)
            Lw = np.dot(self.tow - (1 / (1 + np.exp((y - np.dot(x, self.w.T) - bias)/self.alpha))), x)
            Lb = self.tow - (1 / (1 + np.exp((y - np.dot(x, self.w.T) - bias)/self.alpha)))
            self.w = self.w + self.learning_rate * Lw
            self.b = self.b + self.learning_rate * Lb
            self.learning_rate = self.learning_rate / self.damp_factor


        return self.w, self.b

    def fit_regression(self, data):
        formula=str(data.columns[-1])+' ~ '+" + ".join(data.columns[:-1])
        mod = smf.quantreg(formula,data)
        res = mod.fit(q=self.tow)
        self.b=res.params["Intercept"]
        self.w=np.array(list(res.params)[1:]).reshape(1,-1)

        return self.w, np.array(self.b)

    def predict(self, data):
        Y = data.iloc[:,-1]
        X = data.iloc[:,0:-1]

        # self.w = np.loadtxt(self.w_file).reshape(1, X.shape[1])  # Reading weights and bias
        # self.b = np.loadtxt(self.b_file).reshape(1, 1)
        m = np.dot(X, self.w.T) + self.b
        n = m.reshape(-1, )

        return n


dfX = pd.read_csv('X_VAR3.csv')
dfY = pd.read_csv('Y_VAR3.csv')

dfY = dfY.rename(columns = {"O1":"y1","O2":"y2","O3":"y3" })
data = pd.concat([dfX,dfY["y1"]], axis= 1)
train_data = data[:15000]
test_data = data[15000:]

wb = pd.DataFrame()    
wb['w'] = [np.zeros(dfX.shape[1],)]
wb['b'] = [np.zeros(1,)] 

qr = Online_QuantileSGD(wb['w'][0], wb['b'][0], tow= 0.5,alpha = 0.001,learning_rate = 0.0001, damp_factor = 1)
weight, bias = qr.fit_regression(train_data)
    
qr1 = Online_QuantileSGD(weight, bias, tow= 0.5,alpha = 0.001,learning_rate = 0.0001, damp_factor = 1)
w , b = qr.fit_online(test_data)
print(w,b)
