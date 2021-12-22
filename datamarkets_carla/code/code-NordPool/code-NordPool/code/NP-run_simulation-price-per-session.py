# 2021 
# Carla Goncalves <carla.s.goncalves@inesctec.pt>
# License GPL(>=3)
# This script implements an adaption for the data market proposed
# in the following paper:
# C. Gonçalves, P. Pinson, R.J. Bessa, "Towards data markets in renewable 
# energy forecasting", IEEE Transactions on Sustainable Energy, 
# vol. 12, no. 1, pp. 533-542, Jan. 2021.
#
# Price is updated only after the session closes, i.e. all buyers receive 
# the same price.

from NP_synthetic_main_sim import *
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import LinearRegression
import pandas as pd

# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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
time_ahead = 1

# A.1.1 wind data
wind_df = pd.read_csv('../data/NP-wind-power.csv', parse_dates=['datetime'], index_col=0)
dataframe = pd.concat([wind_df.shift(time_ahead), wind_df], axis=1).fillna(0)
dataframe.fillna(dataframe.mean())
dfX = dataframe.iloc[:, :6]
dfY = dataframe.iloc[:, 6:]

# A.1.1 spot prices
spot_prices = pd.read_csv('../data/NP-spot-prices.csv', parse_dates=['datetime'], index_col=0)
spot_prices.index = pd.DatetimeIndex(spot_prices.index).to_period('H')
spot_prices = spot_prices.fillna(spot_prices.mean())

# A.1.2 up prices
up_prices = pd.read_csv('../data/NP-up-prices.csv', parse_dates=['datetime'], index_col=0)
up_prices.index = pd.DatetimeIndex(up_prices.index).to_period('H')
up_prices = up_prices.fillna(up_prices.mean())

# A.1.3 down prices
down_prices = pd.read_csv('../data/NP-down-prices.csv', parse_dates=['datetime'], index_col=0)
down_prices.index = pd.DatetimeIndex(down_prices.index).to_period('H')
down_prices = down_prices.fillna(down_prices.mean())

# A.2 DEFINE PARAMETERS
window_size = 365 * 24  # the size of the train set
set_hours(24 * 31)  # number of observations to estimate the gain 
steps_t = 1  # how much times ahead the temporal window slides
ndays = 10  # number of times the platform slides the window

buyers_ = spot_prices.columns  # names of buyers
sellers_ = spot_prices.columns  # names of sellers
N = len(buyers_)  # number of buyers 
M = len(sellers_)  # number of sellers
Bmin = 0.2  # minimum possible price
Bmax = 0.8  # maximum possible price
epsilon = 0.05  # increments on price
delta = 0.05  # parameter for updating price weights
bids = np.repeat(0.5, N)  # each buyer values a marginal improvement of
# 1% in NRMSE as 5EUR
max_paym = np.repeat(5000, N)  # maximum value each buyer is able to pay
possible_p = np.arange(Bmin, Bmax + epsilon / 2, epsilon)  # all possible prices
w = np.repeat(1.0, len(possible_p))  # initial weights in price (uniform)

# A.3 SAVE RELEVANT BUYERS/SELLERS INFO
buyers = []  # save buyers information

for i, k in enumerate(buyers_):
    # this cycle processes the csv files. Objective:  define for each agent
    # their own variable, and the ones (s)he can buy
    buyers.append(Buyer())
    Y = dfY[k]
    selfX = dfX[[k]]
    X = dfX[[j for j in dfY.columns if j != k]].copy()
    X[selfX.columns] = selfX

    # imbalance price forecast - linear regression with most recent measurement
    lambda_up = up_prices[k] - spot_prices[k]
    lambda_up[lambda_up < 0] = 0

    lambda_down = spot_prices[k] - down_prices[k]
    lambda_down[lambda_down < 0] = 0

    buyers[i].update_parameters(selfX, X, Y, bids[i], max_paym[i], lambda_up, lambda_down)

# #######################################
# B. DATA MARKET SIMULATION
# #######################################
results = np.zeros((ndays, M + 5, N))  # it will save relevant results

for day in np.arange(0, ndays):  # cycle to simulate the sliding window
    print('>>>\n >>> Session:', day, '\n>>>')
    # 1st step: market sets price before buyers arrive
    if day == 0:
        p = possible_p.mean()  # select the mean of possble prices
        p = (p // epsilon + 1) * (epsilon)
    else:
        p = sum(probs * possible_p)
        p = (p // epsilon + 1) * (epsilon)  # market's price = expected value

    print('1 - Market set price for the entire session:', p)

    for n, k in enumerate(buyers_):  # when a new buyer arrives

        print('###################\n ', dfY.columns[n], '\n###################')

        # 2nd step: Buyer i arrives
        print('2 - Buyer', k, 'arrives')
        b = buyers[n].b  # bid
        # available features for this specific buyer
        X = buyers[n].X.iloc[(day * steps_t):(window_size + day * steps_t), :]
        Y = buyers[n].Y.iloc[(day * steps_t):(window_size + day * steps_t)]

        # 3rd step: Buyer i bids b
        # b = f(...) in this case the same bid is assumed for all buyers
        print('3 - Buyer', k, ' bids', b)

        # 4th step: market allocates features:
        sigma = 0.5 * X.std().mean()
        np.random.seed(1)
        noise = np.random.normal(0, sigma, X.shape)
        Xalloc = data_allocation(p, b, X, noise)

        print('4 - Market allocates features to Buyer', k,
              ' adding noise Normal(0,(sigma x', np.round(max(0, p - b)), '))^2')

        # 5th step: Buyer n computes the gain (now the gain also depends on the electricity market's prices)
        lambda_up = buyers[n].lambda_up[(day * steps_t):(window_size + day * steps_t)]
        lambda_down = buyers[n].lambda_down[(day * steps_t):(window_size + day * steps_t)]
        nominal_leves_all = nominal_level_forecast(lambda_up, lambda_down)
        nominal_levels = nominal_leves_all['pred_nominal_levels_past'].iloc[:, 0]
        g = model(Xalloc, Y, lambda_up, lambda_down, nominal_levels)

        print('5 - Buyer', k, 'had a mean electricity market gain of', g)
        # 6th step: revenue computation
        if b == Bmin:
            r = g * b
            print('here')
        else:
            r = revenue_posAlloc(p, b, Y, X, noise, Bmin, epsilon, lambda_up, lambda_down, nominal_levels)
        print('6 - Market computes the revenue', r)

        results[day, 3, n] = r  # the value paid by the buyer n
        results[day, 4, n] = r  # the value paid by the buyer n

        if r > max_paym[n]:
            r_ = r
            b_candidates = possible_p[possible_p < b]
            i = 0
            while r_ > max_paym[n]:
                i += 1
                r_ = revenue_posAlloc(p, b_candidates[-i], Y, X, noise, Bmin, epsilon, lambda_up, lambda_down,
                                      nominal_levels)
            b = b_candidates[-i]
            Xalloc = data_allocation(p, b, X, noise)  # it is necessary to allocate again
            g = model(Xalloc, Y, lambda_up, lambda_down, nominal_levels)

            results[day, 4, n] = r_  # the value paid by the buyer n due to the max restriction

            print('Bid adjusted because final payment surpasses the maximum. New bid:', b,
                  'Corresponding payment:', r_)

        # 7th step: divide money by sellers
        if r > 0:
            r_division = shapley_robust(Y, Xalloc, 5, 1, lambda_up, lambda_down, nominal_levels)
            results[day, 6:(6 + M - 1), n] = r_division  # money division by sellers
            del r_division

        # 8th step: compute the effective gain when predicting 1h-ahead
        lambda_up = buyers[n].lambda_up.iloc[(window_size + day * steps_t)]
        lambda_down = buyers[n].lambda_down.iloc[(window_size + day * steps_t)]
        nominal_levels = nominal_leves_all['pred_nominal_levels_future'].values[0]

        y_own = QuantReg(Y, X[k].iloc[0:(X.shape[0])]).fit(q=nominal_levels).predict(
            buyers[n].X[k].iloc[(window_size + day * steps_t):(window_size + day * steps_t + 1)])
        g_own = gain(buyers[n].Y[(window_size + day * steps_t)], y_own, lambda_up, lambda_down)

        y_market = QuantReg(Y, X).fit(q=nominal_levels).predict(
            buyers[n].X.iloc[(window_size + day * steps_t), :].values.reshape((1, -1)) + max(0, p - b) * noise[1, :])
        g_market = gain(buyers[n].Y[(window_size + day * steps_t)], y_market, lambda_up, lambda_down)

        rev_purchased_forecast = g_market - g_own
        print('Real gain', n + 1, 'had a gain funtion of', rev_purchased_forecast)

        # save relevant info
        results[day, 0, n] = p  # price fixed by the market for buyer n
        results[day, 1, n] = b  # bid offered by buyer n
        results[day, 2, n] = g * b  # gain estimated by the market for buyer n
        results[day, 5, n] = rev_purchased_forecast  # gain forecasting 1h-ahead
        del r, noise, g

    print('Market closed. Let us update prices for the next round')

    for n, k in enumerate(buyers_):
        # update price weights
        X = buyers[n].X.iloc[(day * steps_t):(window_size + day * steps_t), :]
        Y = buyers[n].Y.iloc[(day * steps_t):(window_size + day * steps_t)]
        lambda_up = buyers[n].lambda_up[(day * steps_t):(window_size + day * steps_t)]
        lambda_down = buyers[n].lambda_down[(day * steps_t):(window_size + day * steps_t)]
        nominal_levels = nominal_level_forecast(lambda_up, lambda_down)['pred_nominal_levels_past'].iloc[:, 0]
        probs, w = price_update(b, Y, X, Bmin, Bmax, epsilon, delta, N, w, lambda_up, lambda_down, nominal_levels)

# save results in 'results' folder
import os

if not os.path.exists('../results'):
    os.makedirs('../results')

p = results[:, 0, :]
b = results[:, 1, :]
g = results[:, 2, :]
r_without_restrictions = results[:, 3, :]
r = results[:, 4, :]
rev_purchased_forecast = results[:, 5, :]
df = pd.DataFrame(b)
df.to_csv(path_or_buf='../results/bids-per-buyer.csv', index=False)
df = pd.DataFrame(p)
df.to_csv(path_or_buf='../results/prices-per-buyer.csv', index=False)
df = pd.DataFrame(g)
df.to_csv(path_or_buf='../results/estimated-gains-per-buyer.csv', index=False)
df = pd.DataFrame(r_without_restrictions)
df.to_csv(path_or_buf='../results/payments-per-buyer-without-max.csv', index=False)
df = pd.DataFrame(r)
df.to_csv(path_or_buf='../results/payments-per-buyer.csv', index=False)
df = pd.DataFrame(rev_purchased_forecast)
df.to_csv(path_or_buf='../results/gains-ahead-per-buyer.csv', index=False)

# compute the values received by seller
rm_all = np.zeros((results.shape[0], results.shape[2]))
for i in np.arange(0, M):
    r_m = 0
    for j in np.arange(0, M):
        if i < j:
            r_m = r_m + results[:, (6 + i), j] * results[:, 4, j]
        if i > j:
            r_m = r_m + results[:, (6 + i - 1), j] * results[:, 4, j]
        rm_all[:, i] = r_m
df = pd.DataFrame(rm_all)
df.to_csv(path_or_buf='../results/revenue-per-seller.csv', index=False)
