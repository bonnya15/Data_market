# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:43:01 2021

@author: ASUS 8I5-8-512-4GTX


"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
#import multiprocessing
from Online_SGD import *

def revenue(p, b, Y,y_own, y_market, Bmin, epsilon):
    # Function that computes the final value to be paid by buyer
    reps = 2 
    expected_revenue = np.repeat(0.0, reps)
    sigma = 0.5*Y.std()
    #print("sigma within revenue func",sigma,"\n")
    #print("price within revenue function=",p,"\n","bid=",b,"\n")
    for i in range(reps):  
        np.random.seed(i)
        noise = np.random.normal(0, sigma, Y.shape)
        def f(z):
            YY = data_allocation(p, z, y_market, noise)
            return gain(Y,y_own, YY)
        xaxis = np.arange(Bmin,b+epsilon,0.5) 
        if len(xaxis)==1:
            expected_revenue[i] = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)))
            #print("gain within function",gain(Y,y_own,data_allocation(p, b, y_market, noise)),"\n")
        else:
            I_ = sum([f(v) for v in xaxis])*(xaxis[1]-xaxis[0])
# =============================================================================
#             for v in xaxis:
#                 print("f(v)=",f(v),"v",v,"\n")
# =============================================================================
            #print('I=',I_,'\n')
            expected_revenue[i] = max(0, b*gain(Y,y_own,data_allocation(p, b, y_market, noise)) - I_)
            #print("gain within function",b*gain(Y,y_own,data_allocation(p, b, y_market, noise)),'\n')
    return expected_revenue.mean()

def data_allocation(p, b, Y, noise):
    # Function which receives the current price (p) and bid (b) and decide
    # the quality at this buyer gets allocate
    Ynoise = Y + max(0, p-b)*0.25*noise
    Ynoise = pd.DataFrame(Ynoise)
    #Y = pd.DataFrame(Y)
    # the last variable is known by the owner!
    #Ynoise.iloc[:, Y.shape[1]-1] = Y.iloc[:, Y.shape[1]-1]
    return Ynoise


def gain(Y,y_own, y_market):
# =============================================================================
#     print(len(Y),len(y_own),len(y_market))
#     print(type(Y),type(y_own),type(y_market))
#     print(y_market)
#     print(y_own)
# =============================================================================
    y_market=y_market.squeeze()
    g_own = np.sqrt(np.mean((Y - y_own)**2))
    g_market = np.sqrt(np.mean((Y - y_market)**2))
    #print("g_own",g_own,"g_market",g_market,"\n")
    g = (g_own-g_market)/(np.max(Y)-np.min(Y)) # gain
    return(max(0,g.mean())*100)  


def revenue_posAlloc(p, b, Y,y_own, y_market, X, noise, Bmin, epsilon):
    # same as revenue but using fixed noise matrix
    def f(z):
        YY = data_allocation(p, z, Y, noise)
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


b,Bmin,epsilon=5,0.5,0.5

x=np.load('a.npy')
y=np.load('b.npy')
z=np.load('c.npy')

#print("normal gain=",gain(x,y,z),"\n")
for p in range(15,105,5):
    print("\np=",p/10)
    print("revenue=",revenue(p/10, b, x,y,z, Bmin, epsilon))