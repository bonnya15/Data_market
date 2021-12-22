

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

lr=0.00001
damp=1.00
# 1. INITIALIZATION

def set_hours(hours):
    global hours_
    hours_ = hours # \Delta parameter -- see paper

class Buyer: # class to save relevant data per buyer
    def update_parameters(self, selfX, X, Y, b, max_paym, coef_market, coef_own,seller):  # Save weights and Biases 
        self.selfX = selfX # own data
        self.Y = Y # data to predict
        self.b = b # bids
        self.max_paym = max_paym # bids
        self.X = X # all available data
        self.coef_market = coef_market #saving the latest coefficients of the model for the market
        self.coef_own = coef_own       #saving the latest coefficients of the model for own data 
        self.seller = seller
        

# 2. RMSE

def RMSE(Y, Yh):  
    Y = pd.DataFrame(Y)
    Yh = pd.DataFrame(Yh)
    rms = np.sqrt(np.mean((Y-Yh)*(Y-Yh))) # RMSE
    return rms[0]

# 3. MODEL USED TO PREDICT BUYER'S DATA

def model(X, Y): # model function (train and evaluate the corresponding gain) 
    # X: covariates
    # Y: target
    X = pd.DataFrame(X)
    # by construction, own features are in the last column
    X_own = X.iloc[:, len(X.columns)-1].copy()
    Y = pd.DataFrame(Y)
    # variables of the owner
    Xown_train = pd.DataFrame(X_own.iloc[0:(X.shape[0]-hours_)])
    Xown_test = pd.DataFrame([X_own.iloc[(X.shape[0]-hours_):]]).T
    # variables from the market
    Xtrain = X.iloc[0:(X.shape[0]-hours_), :]
    Xtest = X.iloc[(X.shape[0]-hours_):, :]
    Ytrain = Y.iloc[0:(X.shape[0]-hours_), :]
    Ytest = Y.iloc[(X.shape[0]-hours_):, :]
    # train models
    model_own = LinearRegression(fit_intercept=True).fit(Xown_train.values, Ytrain.values)
    model_market = LinearRegression(fit_intercept=True).fit(Xtrain.values, Ytrain.values)
    
    #print(X_own)
    weight_market= model_market.coef_.reshape(1, X.shape[1])
    weight_own= model_own.coef_.reshape(1,1)
    bias_market=model_market.intercept_.reshape(1, 1)
    bias_own=model_own.intercept_.reshape(1, 1)
    coeff={'w':[weight_market,weight_own],'b':[bias_market,bias_own]}
    coeff = pd.DataFrame(coeff)    
    
    y_market = []
    y_own = []
    y = []
    
    y.append(Ytest[(Ytest.shape[0]-hours_):(Ytest.shape[0]-hours_+1)][0][8016])
    y_market.append(model_market.predict(X[(X.shape[0]-hours_):(X.shape[0]-hours_+1)])[0][0])
    y_own.append(model_own.predict(X_own[(X.shape[0]-hours_):(X.shape[0]-hours_+1)].values.reshape((1,-1)))[0][0])

    for i in range(hours_):
        
        
        model_market_online = Online_SGD(coeff[0:1]['w'][0], coeff[0:1]['b'][0], learning_rate=lr,damp_factor=damp)
        weight_market, bias_market =  model_market_online.fit_online(Xtest[i:i+1], Ytest[i:i+1])
    
        model_own_online = Online_SGD(coeff[1:2]['w'][1], coeff[1:2]['b'][1], learning_rate=lr,damp_factor=damp)
        weight_own , bias_own = model_own_online.fit_online(Xown_test[i:i+1], Ytrain[i:i+1])
        
        coeff={'w':[weight_market,weight_own],'b':[bias_market,bias_own]}
        coeff = pd.DataFrame(coeff)
        
        if i == hours_-1 :
            break 
        else:
            y.append(Ytest[i+1:i+2].values[0][0])
            y_market.append(model_market_online.predict(Xtest[i+1:i+2])[0])
            y_own.append(model_own_online.predict(Xown_test[i+1:i+2])[0])        
        
        

    df = pd.DataFrame(columns=('Y','y_own','y_market'))
    df['Y'] = y
    df['y_own'] = y_own
    df['y_market'] = y_market

    return (pd.DataFrame(coeff),df)




def model_Online(X,Y, wb_market, wb_own):
    # X: covariates
    # Y: target
    # by construction, own features are in the last column
    X_own = X.iloc[:, len(X.columns)-1].copy()
    X_own = X_own.values.reshape(-1, 1)
    Y = pd.DataFrame(Y)
# =============================================================================
#     y_market = []
#     y_own = []
#     y = []
# =============================================================================

    # train models
    model_market_online = Online_SGD(wb_market['w'][0], wb_market['b'][0], learning_rate=lr,damp_factor=damp)
    weight_market, bias_market =  model_market_online.fit_online(X, Y)
    
    model_own_online = Online_SGD(wb_own['w'][0], wb_own['b'][0], learning_rate=lr,damp_factor=damp)
    weight_own , bias_own = model_own_online.fit_online(X_own, Y)
    

    coeff= {'w' : [weight_market, weight_own], 'b' : [bias_market, bias_own]}
# =============================================================================
#     y.append(Y[i+1:i+2].values)
#     y_market.append(model_market_online.predict(Xtest[i+1:i+2]))
#     y_own.append(model_own_online.predict(Xown_test[i+1:i+2]))
#     
#     df = pd.DataFrame(columns=('Y','y_own','y_market'))
#     df['Y'] = y
#     df['y_own'] = y_own
#     df['y_market'] = y_market
# =============================================================================

    return (pd.DataFrame(coeff))     

    
def gain(Y,y_own, y_market):
# =============================================================================
#     print(len(Y),len(y_own),len(y_market))
#     print(type(Y),type(y_own),type(y_market))
#     print(y_market)
#     print(y_own)
# =============================================================================
    y_market=y_market.squeeze()
    g_own = np.sqrt(np.mean((Y.values - y_own)**2))
    g_market = np.sqrt(np.mean((Y.values - y_market)**2))
    g = (g_own-g_market)/(np.max(Y)-np.min(Y)) # gain



    return(max(0,g.mean())*100)    
        
    

# 4. DATA ALLOCATION - PAPER'S EQUATION (18)

def data_allocation(p, b, Y, noise):
    # Function which receives the current price (p) and bid (b) and decide
    # the quality at this buyer gets allocate
    Ynoise = Y + max(0, p-b)*0.25*noise
    Ynoise = pd.DataFrame(Ynoise)
    #Y = pd.DataFrame(Y)
    # the last variable is known by the owner!
    #Ynoise.iloc[:, Y.shape[1]-1] = Y.iloc[:, Y.shape[1]-1]
    return Ynoise

# 5. REVENUE - PAPER'S EQUATION (19)

def revenue(p, b, Y,y_own, y_market, X, Bmin, epsilon):
    # Function that computes the final value to be paid by buyer
    reps = 10
    expected_revenue = np.repeat(0.0, reps)
    sigma = 0.4*Y.std()
# =============================================================================
#     print("sigma within revenue func",sigma,"\n")
#     print("price within revenue function=",p,"\n","bid=",b,"\n")
# =============================================================================
    for i in range(reps):  
        np.random.seed(i)
        noise = np.random.normal(0, sigma, Y.shape)
        def f(z):
            YY = data_allocation(p, z, y_market, noise)
            return gain(Y,y_own, YY)
        #Yalloc=data_allocation(p, b, Y, noise)
        xaxis = np.arange(Bmin,b+epsilon,0.5) 
        if len(xaxis)==1:
            expected_revenue[i] = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)))
            #print("gain within function",gain(Y,y_own,data_allocation(p, b, y_market, noise)),"\n")
        else:
            I_ = sum([f(v) for v in xaxis])*(xaxis[1]-xaxis[0])
            #print('I=',I_,'\n')
            expected_revenue[i] = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)) - I_)
            #print("gain within function",gain(Y,y_own,data_allocation(p, b, y_market, noise)),'\n')
    #print("price=",p," bid=", b," revenue func expected revenue mean",expected_revenue.mean())
    return expected_revenue.mean()


def revenue_posAlloc(p, b, Y,y_own, y_market, X, noise, Bmin, epsilon):
    # same as revenue but using fixed noise matrix
    def f(z):
        YY = data_allocation(p, z, y_market, noise)
        return gain(Y,y_own, YY)
    xaxis = np.arange(Bmin,b+epsilon,0.5) 
    if len(xaxis)==1:
        expected_revenue = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)))
        #print("gain within function",gain(Y,y_own,data_allocation(p, b, y_market, noise)),'\n')
    else:
        I_ = sum([f(v) for v in xaxis])*(xaxis[1]-xaxis[0])
        #print('I=',I_,'\n')
        expected_revenue = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)) - I_)
        #print("gain within function",gain(Y,y_own,data_allocation(p, b, y_market, noise)),'\n')

    return expected_revenue

# 6. PRICE UPDATE - PAPER'S ALGORITHM 2

def aux_price(c_, w_last, b, Y,y_own, y_market, X, Bmax, delta, Bmin, epsilon):
    g = revenue(c_, b, Y,y_own, y_market, X, Bmin, epsilon)/Bmax
    w = (1-delta)*w_last+delta*g
    return w
    
def price_update(b, Y,y_own, y_market, X, Bmin, Bmax, epsilon, delta, N, w):
    c = np.arange(Bmin, Bmax+epsilon, epsilon)
    #print("weights before",w,"\n")
    Wn = np.sum(w)
    probs = w/Wn
    res = []
    for j, c_ in enumerate(c):
       res.append(aux_price(c_, w[j], b, Y,y_own, y_market, X, Bmax, delta, Bmin, epsilon))
    w = np.array(res)
    w = w.transpose()
    #print("weights after",w,"\n")
    # print('weights', w/np.sum(w))
    Wn = np.sum(w) # this line was missing
    probs = w/Wn
    return probs, w

def gain_paydiv(Y,y_inc, y_exc):

    y_inc=y_inc.squeeze()
    g_exc = np.sqrt(np.mean((Y.values - y_exc)**2))
    g_inc = np.sqrt(np.mean((Y.values - y_inc)**2))
    g = (g_inc-g_exc) # gain

    return(max(0,g.mean())*100)   

# 7. PAYMENT DIVISION - PAPER'S ALGORITHM 1

def aux_shap_aprox(m, M, K, X, Y):
    phi_ = 0
    new = pd.DataFrame(columns = ['exclude_w','exclude_b','include_w','include_b'])
    new['exclude_w'] = [[np.zeros(2,)] for i in range(K)]
    new['exclude_b'] = [[np.zeros(1,)] for i in range(K)]
    new['include_w'] = [[np.zeros(2,)] for i in range(K)]
    new['include_b'] = [[np.zeros(1,)] for i in range(K)]

    
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
        XX[X.columns[M]] = X[X.columns[M]]  # exclude own seller data
        coeff,df = model(XX, Y)
        y_pred = pd.DataFrame(columns=('Y','exclude_seller','include_seller'))
        y_pred['Y'] = df['Y'].values
        y_pred['exclude_seller'] = df['y_market'].values
        
        new['exclude_w'][k] = coeff['w'][0]
        new['exclude_b'][k] = coeff['b'][0]

        
        pos = sg[np.arange(0, np.where(sg == m)[0][0]+1)]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        coeff,df = model(XX, Y)
        y_pred['include_seller'] = df['y_market'].values
        
        
        
        G = gain_paydiv(y_pred['Y'], y_pred['include_seller'],y_pred['exclude_seller'])
        
        new['include_w'][k] = coeff['w'][0]
        new['include_b'][k] = coeff['b'][0]
        
        phi_ += max(0, G)
    print(phi_/K)
    return(phi_, new)


class Sellers: # class to save relevant data per buyer
    def update_parameters(self, coef_df,gain):  # Save weights and Biases 
        self.coef_df = coef_df
        self.gain = gain


def shapley_aprox(Y, X, K):
    M = X.shape[1]-1
    phi = np.repeat(0.0, M)
    res = []
    seller = []
    for i in range(M):
        seller.append(Sellers())
    for m in np.arange(0, M):
        r , df = aux_shap_aprox(m, M, K, X, Y)
        seller[m].update_parameters(df,r/K)
        res.append(r)
    phi = np.array([r for r in res])
    phi = phi.transpose()
    return(phi/K, seller)


def square_rooted(x):
    return np.round(np.sqrt(sum([a*a for a in x])), 3)


def cos_similarity(x, y):
    numerator = sum(a*b for a, b in zip(x, y))
    denominator = square_rooted(x)*square_rooted(y)
    return np.round(np.abs(numerator)/np.float(denominator), 3)


def shapley_robust(Y, X, K, lambd):
    M = X.shape[1]-1
    phi_ = np.repeat(0.0, M)
    phi,seller_class  = shapley_aprox(Y, X, K)
    for m in np.arange(0, M):
        s = 0
        for k in np.arange(0, M):
            if k != m:
                s += cos_similarity(X.iloc[:, m], X.iloc[:, k])
        phi_[m] = phi[m]*np.exp(-lambd * s)
    if phi.sum()>0:
        phi = phi_/phi_.sum() 
    return(phi,seller_class)






## Profit division algorithm online

def shapley_robust_online(Y, X, K, lambd, seller_list):
    M = X.shape[1]-1
    phi_ = np.repeat(0.0, M)
    phi,seller_class = shapley_aprox_online(Y, X, K, seller_list)
    for m in np.arange(0, M):
        s = 0
        for k in np.arange(0, M):
            if k != m:
                s += cos_similarity(X.iloc[:, m], X.iloc[:, k])
        phi_[m] = phi[m]*np.exp(-lambd * s)
    if phi.sum()>0:
        phi = phi_/phi_.sum() 
    return(phi,seller_class)

def shapley_aprox_online(Y, X, K,seller_list):
    M = X.shape[1]-1
    phi = np.repeat(0.0, M)
    res = []
    seller = seller_list

    for m in np.arange(0, M):
        r , df = aux_shap_aprox_online(m, M, K, X, Y,seller[m].coef_df, seller[m].gain, delta = (1/744))
        seller[m].update_parameters(df,r)
        res.append(r)
    phi = np.array([r for r in res])
    phi = phi.transpose()
    return(phi, seller)

def aux_shap_aprox_online(m, M, K, X, Y,df,past_gain,delta):
    phi_ = 0
    ## Coefficient DataFrames
    wb_exclude = df[['exclude_w','exclude_b']].rename({'exclude_w': 'w', 'exclude_b': 'b'}, axis='columns').copy()
    wb_include = df[['include_w','include_b']].rename({'include_w': 'w', 'include_b': 'b'}, axis='columns').copy()
    wb_own = pd.DataFrame()
    wb_own['w'] = [[np.zeros(1,)]] 
    wb_own['b'] = [[np.zeros(1,)]]  
    
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
        XX[X.columns[M]] = X[X.columns[M]]  # exclude own seller data
        coeff = model_Online(XX, Y, wb_exclude.loc[k], wb_own.loc[0])
        
        ## Online Prediction part
        market = Online_SGD(wb_exclude['w'][k], wb_exclude['b'][k], learning_rate=lr,damp_factor=damp)
        y_market = market.predict(XX)
            
        y_pred = pd.DataFrame(columns=('Y','exclude_seller','include_seller'))
        y_pred['Y'] = Y[0]
        y_pred['exclude_seller'] = y_market
        
        
        df['exclude_w'][k] = coeff['w'][0]
        df['exclude_b'][k] = coeff['b'][0]
        
        
        pos = sg[np.arange(0, np.where(sg == m)[0][0]+1)]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        coeff = model_Online(XX, Y, wb_include.loc[k], wb_own.loc[0])
        market = Online_SGD(wb_include['w'][k], wb_include['b'][k], learning_rate=lr,damp_factor=damp)
        y_market = market.predict(XX)
        
        y_pred['include_seller'] = y_market
        G = gain_paydiv(y_pred['Y'], y_pred['include_seller'],y_pred['exclude_seller'])
       
        

        
        df['include_w'][k] = coeff['w'][0]
        df['include_b'][k] = coeff['b'][0]

        phi_ += max(0, G)
    
    
    new_G = (1-delta)*past_gain + delta*(phi_/K)
    print(phi_/K, new_G)
    return(new_G, df)



