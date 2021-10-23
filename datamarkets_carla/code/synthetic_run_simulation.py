# 2020 
# Carla Goncalves <carla.s.goncalves@inesctec.pt>
# License GPL(>=3)
# This script implements the main functions to simulate the data market proposed
# in the following paper:
# C. Gonçalves, P. Pinson, R.J. Bessa, "Towards data markets in renewable 
# energy forecasting", IEEE Transactions on Sustainable Energy, 
# vol. 12, no. 1, pp. 533-542, Jan. 2021.

from synthetic_main_sim import *
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# ---------------------------------------------------------------------------#
# ----------------------------- SCRIPT CONTENTS -----------------------------#
# ---------------------------------------------------------------------------#
# A. INITIALIZATION
# A.1 READ DATA
# A.2 DEFINE PARAMETERS
# A.3 SAVE RELEVANT BUYERS/SELLERS INFO
# B. DATA MARKET SIMULATION 
# ---------------------------------------------------------------------------#

# #######################################
# A. INITIALIZATION 
# #######################################

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
results = np.zeros((ndays, M+5, N))  # it will save relevant results

for day in np.arange(0,ndays): # cycle to simulate the sliding window
    print('>>>\n >>> Day:', day, '\n>>>')
    for n, k in enumerate(buyers_): # when a new buyer arrives
        
        print('###################\n ', dfY.columns[k],'\n###################')
              
        # 1st step: market sets price
        if (day == 0) & (n==0): 
            p = np.random.uniform(Bmin, Bmax) # select a random price
        else:
            # NEW - define the price as the mean value of the distribution:
            p = sum(probs*possible_p) 
            p = (p//epsilon+1)*(epsilon) # market's price = expected value
            # OLD - The paper considers the price by random generation: 
            # U = np.random.uniform(0, 1)
            # p = possible_p[np.max(np.where(U<=(np.cumsum(probs)))[0][0])]
        print('1 - Market set price:', p)
    
        # 2nd step: Buyer i arrives
        print('2 - Buyer', n+1, 'arrives')
        b = buyers[n].b # bid         
        # available features for this specific buyer
        X = buyers[n].X.iloc[(day*steps_t):(window_size+day*steps_t), :] 
        Y = buyers[n].Y[(day*steps_t):(window_size+day*steps_t)]
        
        # 3rd step: Buyer i bids b
        # b = f(...) in this case the same bid is assumed for all buyers
        print('3 - Buyer', n+1, ' bids', b)
        
        # 4th step: market allocates features:
        sigma = 0.5*X.std().mean()
        noise = np.random.normal(0, sigma, X.shape)
        Xalloc = data_allocation(p, b, X, noise)
        
        print('4 - Market allocates features to Buyer', n+1,
              ' adding noise Normal(0,(sigma x', np.round(max(0, p-b)), '))^2')
                
        # update price weights  
        probs, w = price_update(b, Y, X, Bmin,Bmax, epsilon, delta, N, w)
        
        # 5th step: Buyer n computes the gain
        g = model(Xalloc.values, Y)
        
        print('5 - Buyer', n+1, 'had a RMSE gain of', g)
        # 6th step: revenue computation
        if b==Bmin: 
            r = g*b
        else:
            r = revenue_posAlloc(p, b, Y, X, noise, Bmin, epsilon)
        print('6 - Market computes the revenue', r)
        
        # 7th step: divide money by sellers
        if r>0:
            r_division = shapley_robust(Y, Xalloc, 5, 1)
            results[day, 5:(5+M-1), n] = r_division # money division by sellers
            del r_division
            
        # 8th step: compute the effective gain when predicting 1h-ahead
        model_market = LinearRegression(fit_intercept=True).fit(Xalloc[0:(X.shape[0])], Y[0:(X.shape[0])])
        y_market = model_market.predict(buyers[n].X.iloc[(window_size+day*steps_t),:].values.reshape((1,-1))+max(0,p-b)*noise[1,:]) 
        y_real = buyers[n].Y[(window_size+day*steps_t)]
        g_market =  RMSE(y_real, y_market)
        
        model_own = LinearRegression(fit_intercept=True).fit(Xalloc.iloc[0:(X.shape[0]), (X.shape[1]-1)].values.reshape(-1, 1), Y[0:(X.shape[0])])
        y_own = model_own.predict(buyers[n].X.iloc[(window_size+day*steps_t):(window_size+day*steps_t+1),(X.shape[1]-1):])
        g_own =  RMSE(y_real, y_own)
        
        rev_purchased_forecast = b*((g_own-g_market)/(np.max(Y)-np.min(Y)))*100
        print('Real gain', n+1, 'had a gain funtion of', rev_purchased_forecast)
        
        # save relevant info
        results[day, 0, n] = p # price fixed by the market for buyer n
        results[day, 1, n] = b # bid offered by buyer n
        results[day, 2, n] = g*b # gain estimated by the market for buyer n
        results[day, 3, n] = r # the value paid by the buyer n
        results[day, 4, n] = rev_purchased_forecast # gain forecasting 1h-ahead
        del X, Y, p, r, noise, g
    #np.save('market_results.npy', results[0:day,:,:])


# save results in 'results' folder
import os
if not os.path.exists('../results'):
    os.makedirs('../results')
    
p = results[:, 0,  :]
b = results[:, 1,  :]
g = results[:, 2,  :]
r = results[:, 3,  :]
rev_purchased_forecast = results[:, 4,  :]
df = pd.DataFrame(b)
df.to_csv(path_or_buf='../results/bids-per-buyer.csv', index=False)
df = pd.DataFrame(p)
df.to_csv(path_or_buf='../results/prices-per-buyer.csv', index=False)
df = pd.DataFrame(g)
df.to_csv(path_or_buf='../results/estimated-gains-per-buyer.csv', index=False)
df = pd.DataFrame(r)
df.to_csv(path_or_buf='../results/payments-per-buyer.csv', index=False)
df = pd.DataFrame(rev_purchased_forecast)
df.to_csv(path_or_buf='../results/gains-ahead-per-buyer.csv', index=False)

# compute the values received by seller
rm_all = np.zeros((results.shape[0], results.shape[2]))
for i in np.arange(0,M):
    r_m = 0
    for j in np.arange(0,M):
        if i<j:
            r_m = r_m + results[:, (5+i), j]*results[:, 3, j]
        if i>j:
            r_m = r_m + results[:, (5+i-1), j]*results[:, 3, j]
        rm_all[:,i] = r_m
df = pd.DataFrame(rm_all)
df.to_csv(path_or_buf='../results/revenue-per-seller.csv', index=False)
