# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:13:34 2021

@author: shiuli Subhra Ghosh
"""

from synthetic_main_sim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
from Online_SGD import *



# A.1 READ DATA
dfX = pd.read_csv('../data/X_VAR3.csv')
dfY = pd.read_csv('../data/Y_VAR3.csv')

# A.2 DEFINE PARAMETERS
window_size = 365*24 # the size of the train set
set_hours(24*31) # number of observations to estimate the gain (used to estimate
# the value to be paid) - the paper's notation is \Delta
steps_t = 1 # how much times ahead the temporal window slides
ndays = 5  # number of times the platform slides the window

buyers_ = np.arange(dfY.shape[1]) # number of buyers
sellers_ = np.arange(dfY.shape[1]) # number of sellers
N = len(buyers_)  # dfY.shape[1]
M = len(sellers_)  # dfX.shape[1]
Bmin = 0.5 # minimum possible price
Bmax = 10 # maximum possible price
epsilon = 0.5 # increments on price
delta = 0.5 # parameter for updating price weights
bids = np.repeat(5, N) # each buyer values a marginal improvement of 
# 1% in NRMSE as 5EUR
possible_p = np.arange(Bmin, Bmax+epsilon, epsilon)  # all possible prices
w = np.repeat(1.0, len(possible_p))  # initial weights in price (uniform)

# A.3 SAVE RELEVANT BUYERS/SELLERS INFO
buyers = [] # save buyers information

for i, k in enumerate(buyers_):
    # this cycle processes the csv file. Objective:  defines for each agent
    # their own variable, and the ones (s)he can buy
    buyers.append(Buyer())
    Y = dfY.iloc[:, [j for j, c in enumerate(dfY.columns) if j == k]]
    Y = np.array(Y, ndmin=2)
    selfX = dfX.iloc[:, [j for j, c in enumerate(dfY.columns) if j == k]]
    X = dfX.iloc[:, [j for j, c in enumerate(dfY.columns) if j != k]].copy()
    X[selfX.columns] = selfX
    max_paym = 10
    buyers[i].update_parameters(selfX, X, Y, bids[i],max_paym)

# #######################################
# B. DATA MARKET SIMULATION 
# #######################################

wb_market = pd.DataFrame()
wb_own = pd.DataFrame()

wb_market['Buyers'] = buyers_
wb_own['Own'] = buyers_
wb_market['w'] = [[np.zeros(len(buyers_,))] for x in range(len(buyers_))]
wb_own['w'] = [[np.zeros(1,)] for x in range(len(buyers_))]
wb_market['b'] = [[np.zeros(1,)] for x in range(len(buyers_))]
wb_own['b'] = [[np.zeros(1,)] for x in range(len(buyers_))]



results = np.zeros((ndays, M+5, N))  # it will save relevant results

new1 = pd.DataFrame(columns = ['Buyer','day','y_market','y_real'])
new2 = pd.DataFrame(columns = ['Buyer','day','y_own','y_real'])

new_market = pd.DataFrame(columns = ['Buyer','day','y_market','y_real'])
new_own = pd.DataFrame(columns = ['Buyer','day','y_own','y_real']) 

for day in np.arange(0,ndays): # cycle to simulate the sliding window
    print('>>>\n >>> Day:', day, '\n>>>')
    for n, k in enumerate(buyers_): # when a new buyer arrives
        
        # 2nd step: Buyer i arrives
        print('2 - Buyer', n+1, 'arrives','in day' , day)
        b = buyers[n].b # bid         
        # available features for this specific buyer
         ## X = buyers[n].X.iloc[(day*steps_t):(window_size+day*steps_t), :] 
         ## Y = buyers[n].Y[(day*steps_t):(window_size+day*steps_t)]
        
        if day == 0:
            
             X = buyers[n].X.iloc[0:(window_size+day*steps_t), :] 
             Y = buyers[n].Y[0 :(window_size+day*steps_t)]
        else:
             X = buyers[n].X.iloc[(window_size+day*steps_t): (window_size+day*steps_t+1),:]
             Y = buyers[n].Y[(window_size+day*steps_t): (window_size+day*steps_t+1)]
             

            
            
        if day == 0 :
            
            model_market = Online_SGD(wb_market['w'][n][0], wb_market['b'][n][0], learning_rate=0.2,damp_factor=1.02)
            wb_market['w'][n] , wb_market['b'][n] = model_market.fit_regression(X[0:(X.shape[0])], Y[0:(X.shape[0])])
            y_market = model_market.predict(buyers[n].X.iloc[(window_size+day*steps_t),:].values.reshape((1,-1)))
            y_real = buyers[n].Y[(window_size+day*steps_t)]
            g_market =  RMSE(y_real, y_market)
        
        
            model_own = Online_SGD(wb_own['w'][n][0], wb_own['b'][n][0], learning_rate=0.2,damp_factor=1.02)
            wb_own['w'][n] , wb_own['b'][n] = model_own.fit_regression(X.iloc[0:(X.shape[0]), (X.shape[1]-1)].values.reshape(-1, 1), Y[0:(X.shape[0])])    
            y_own = model_own.predict(buyers[n].X.iloc[(window_size+day*steps_t):(window_size+day*steps_t+1),(X.shape[1]-1):])    
            g_own =  RMSE(y_real, y_own)

            
        else:
            model_market = Online_SGD(wb_market['w'][n][0], wb_market['b'][n][0], learning_rate=0.2,damp_factor=1.02)
            wb_market['w'][n] , wb_market['b'][n] = model_market.fit_online(X[0:(X.shape[0])], Y[0:(X.shape[0])])
            y_market = model_market.predict(buyers[n].X.iloc[(window_size+day*steps_t),:].values.reshape((1,-1)))
            y_real = buyers[n].Y[(window_size+day*steps_t)]
            g_market =  RMSE(y_real, y_market)
        
            model_own = Online_SGD(wb_own['w'][n], wb_own['b'][n], learning_rate=0.2,damp_factor=1.02)
            wb_own['w'][n] , wb_own['b'][n] = model_own.fit_online(X.iloc[0:(X.shape[0]), (X.shape[1]-1)].values.reshape(-1, 1), Y[0:(X.shape[0])])    
            y_own = model_own.predict(buyers[n].X.iloc[(window_size+day*steps_t):(window_size+day*steps_t+1),(X.shape[1]-1):])    
            g_own =  RMSE(y_real, y_own)  

            
        new1['Buyer'] = n+1
        new1['day'] = day+1
        new1['y_market'] = y_market
        new1['y_real'] = y_real 
        
        new2['Buyer'] = n+1
        new2['day'] = day+1
        new2['y_own'] = y_own
        new2['y_real'] = y_real
        
        new_market = new_market.append(new1, ignore_index = True)
        new_own = new_own.append(new2, ignore_index = True)  
            

            
 