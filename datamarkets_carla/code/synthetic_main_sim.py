# 2020 
# Carla Goncalves <carla.s.goncalves@inesctec.pt>
# License GPL(>=3)
# This script simulates the data market for the synthetic experiments in
# C. Gonçalves, P. Pinson, R.J. Bessa, "Towards data markets in renewable 
# energy forecasting", IEEE Transactions on Sustainable Energy, 
# vol. 12, no. 1, pp. 533-542, Jan. 2021.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
#import multiprocessing
from Online_SGD import *

# ---------------------------------------------------------------------------#
# ----------------------------- SCRIPT CONTENTS -----------------------------#
# ---------------------------------------------------------------------------#
# 1. INITIALIZATION
# 2. RMSE
# 3. MODEL USED TO PREDICT BUYER'S DATA
# 4. DATA ALLOCATION - PAPER'S EQUATION (18)
# 5. REVENUE - PAPER'S EQUATION (19)
# 6. PRICE UPDATE - PAPER'S ALGORITHM 2
# 7. PAYMENT DIVISION - PAPER'S ALGORITHM 1
# ---------------------------------------------------------------------------#

## Incorporating online SGD


# 1. INITIALIZATION

def set_hours(hours):
    global hours_
    hours_ = hours # \Delta parameter -- see paper

class Buyer: # class to save relevant data per buyer
    def update_parameters(self, selfX, X, Y, b, max_paym):
        self.selfX = selfX # own data
        self.Y = Y # data to predict
        self.b = b # bids
        self.max_paym = max_paym # bids
        self.X = X # all available data

# 2. RMSE

def RMSE(Y, Yh):  
    Y = pd.DataFrame(Y)
    Yh = pd.DataFrame(Yh)
    rms = np.sqrt(np.mean((Y-Yh)*(Y-Yh))) # RMSE
    return rms[0]

# 3. MODEL USED TO PREDICT BUYER'S DATA

def model(X, Y, weight, bias): # model function (train and evaluate the corresponding gain) 
    # X: covariates
    # Y: target
    X = pd.DataFrame(X)
    # by construction, own features are in the last column
    X_own = X.iloc[:, len(X.columns)-1].copy()
    Y = pd.DataFrame(Y)
    # variables of the owner
    Xown_train = pd.DataFrame(X_own.iloc[0:(X.shape[0]-hours_-1)])
    Xown_test = pd.DataFrame([X_own.iloc[(X.shape[0]-hours_):]])
    # variables from the market
    Xtrain = X.iloc[0:(X.shape[0]-hours_-1), :]
    Xtest = X.iloc[(X.shape[0]-hours_):, :]
    Ytrain = Y.iloc[0:(X.shape[0]-hours_-1), :]
    Ytest = Y.iloc[(X.shape[0]-hours_):, :]
    # train models
    #model_own = LinearRegression(fit_intercept=True).fit(Xown_train.values, Ytrain.values)
    #model_market = LinearRegression(fit_intercept=True).fit(Xtrain.values, Ytrain.values)

    model_own = Online_SGD(weight, bias, learning_rate=0.2,damp_factor=1.02)
    w1,b1 = model_own.fit_regression(Xown_train.values, Ytrain.values)
    model_market = Online_SGD(weight, bias, learning_rate=0.2,damp_factor=1.02)
    w2,b2 = model_market.fit_regression(Xtrain.values, Ytrain.values)

    # compute the rmse for both models
    y_own =  model_own.predict(Xown_test.values.reshape((hours_,1)))
    g_own = np.sqrt(np.mean((Ytest.values - y_own)**2))
    y_market = model_market.predict(Xtest.values)
    g_market = np.sqrt(np.mean((Ytest.values - y_market)**2))
    g = (g_own-g_market)/(np.max(Y)-np.min(Y)) # gain

    return (max(0,g.mean())*100) 

# 4. DATA ALLOCATION - PAPER'S EQUATION (18)

def data_allocation(p, b, X, noise):
    # Function which receives the current price (p) and bid (b) and decide
    # the quality at this buyer gets allocate
    Xnoise = X + max(0, p-b)*noise
    Xnoise = pd.DataFrame(Xnoise)
    X = pd.DataFrame(X)
    # the last variable is known by the owner!
    Xnoise.iloc[:, X.shape[1]-1] = X.iloc[:, X.shape[1]-1]
    return Xnoise

# 5. REVENUE - PAPER'S EQUATION (19)

def revenue(p, b, Y, X, Bmin, epsilon):
    # Function that computes the final value to be paid by buyer
    reps = 5 
    expected_revenue = np.repeat(0.0, reps)
    sigma = X.std().mean()
    for i in range(reps):  
        np.random.seed(i)
        noise = np.random.normal(0, sigma, X.shape)
        def f(z):
            XX = data_allocation(p, z, X, noise)
            return model(XX, Y)
        X_alloc = data_allocation(p, b, X, noise)
        xaxis = np.arange(Bmin,b+epsilon,0.5) 
        if len(xaxis)==1:
            expected_revenue[i] = max(0, b*model(X_alloc, Y))
        else:
            I_ = sum([f(v) for v in xaxis])*(xaxis[1]-xaxis[0])
            expected_revenue[i] = max(0, b*model(X_alloc, Y) - I_)
    return expected_revenue.mean()


def revenue_posAlloc(p, b, Y, X, noise, Bmin, epsilon):
    # same as revenue but using fixed noise matrix
    def f(z):
        XX = data_allocation(p, z, X, noise)
        return model(XX, Y)
    X_alloc = data_allocation(p, b, X, noise)
    xaxis = np.arange(Bmin,b+epsilon,0.5) 
    if len(xaxis)==1:
        expected_revenue = max(0, b*model(X_alloc, Y))
    else:
        I_ = sum([f(v) for v in xaxis])*(xaxis[1]-xaxis[0])
        expected_revenue = max(0, b*model(X_alloc, Y) - I_)

    return expected_revenue

# 6. PRICE UPDATE - PAPER'S ALGORITHM 2

def aux_price(c_, w_last, b, Y, X, Bmax, delta, Bmin, epsilon):
    g = revenue(c_, b, Y, X, Bmin, epsilon)/Bmax
    w = (1-delta)*w_last+delta*g
    return w
    
def price_update(b, Y, X, Bmin, Bmax, epsilon, delta, N, w):
    c = np.arange(Bmin, Bmax+epsilon, epsilon)
    Wn = np.sum(w)
    probs = w/Wn
    res = []
    for j, c_ in enumerate(c):
       res.append(aux_price(c_, w[j], b, Y, X, Bmax, delta, Bmin, epsilon))
    #w = np.array([r.get() for r in res])
    w = w.transpose()
    # print('weights', w/np.sum(w))
    Wn = np.sum(w) # this line was missing
    probs = w/Wn
    return probs, w

# 7. PAYMENT DIVISION - PAPER'S ALGORITHM 1

def aux_shap_aprox(m, M, K, X, Y):
    phi_ = 0
    for k in np.arange(0, K):
        np.random.seed(k)
        sg = np.random.permutation(M)
        i = 1
        while sg[0] == m:
            np.random.seed(i)
            sg = np.random.permutation(M)
            i+=1
        pos = sg[np.arange(0, np.where(sg == m)[0][0])]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        G = model(XX, Y)
        pos = sg[np.arange(0, np.where(sg == m)[0][0]+1)]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        Gplus = model(XX, Y)
        phi_ += max(0, Gplus-G)
    return phi_


def shapley_aprox(Y, X, K):
    M = X.shape[1]-1
    phi = np.repeat(0.0, M)
    res = []
    for m in np.arange(0, M):
       res.append(aux_shap_aprox(m, M, K, X, Y))
    #phi = np.array([r.get() for r in res])
    phi = phi.transpose()
    return phi/K


def square_rooted(x):
    return np.round(np.sqrt(sum([a*a for a in x])), 3)


def cos_similarity(x, y):
    numerator = sum(a*b for a, b in zip(x, y))
    denominator = square_rooted(x)*square_rooted(y)
    return np.round(np.abs(numerator)/np.float(denominator), 3)


def shapley_robust(Y, X, K, lambd):
    M = X.shape[1]-1
    phi_ = np.repeat(0.0, M)
    phi = shapley_aprox(Y, X, K)
    for m in np.arange(0, M):
        s = 0
        for k in np.arange(0, M):
            if k != m:
                s += cos_similarity(X.iloc[:, m], X.iloc[:, k])
        phi_[m] = phi[m]*np.exp(-lambd * s)
    if phi.sum()>0:
        phi = phi_/phi_.sum() 
    return phi
