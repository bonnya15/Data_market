# 2020 
# Carla Goncalves <carla.s.goncalves@inesctec.pt>
# License GPL(>=3)
# This script simulates the data market for the synthetic experiments in
# C. Gonçalves, P. Pinson, R.J. Bessa, "Towards data markets in renewable 
# energy forecasting", IEEE Transactions on Sustainable Energy, 
# vol. 12, no. 1, pp. 533-542, Jan. 2021.

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd
import multiprocessing


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


# 1. INITIALIZATION

def set_hours(hours):
    global hours_
    hours_ = hours  # \Delta parameter -- see paper


class Buyer:  # class to save relevant data per buyer
    def update_parameters(self, selfX, X, Y, b, max_paym, lambda_up, lambda_down):
        self.selfX = selfX  # own data
        self.Y = Y  # data to predict
        self.b = b  # bids
        self.max_paym = max_paym  # bids
        self.X = X  # all available data
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down


# 2.1 RMSE

def RMSE(Y, Yh):
    Y = pd.DataFrame(Y)
    Yh = pd.DataFrame(Yh)
    rms = np.sqrt(np.mean((Y - Yh) * (Y - Yh)))  # RMSE
    return rms[0]


# 2.2 Electricity market costs

def gain(Y, Yh, lambda_up, lambda_down):  # gain just consider the performance on the test! Expression (2) of the paper.
    residuals = Yh - Y
    max_residuals = residuals.copy()
    max_residuals[max_residuals < 0] = 0
    min_residuals = residuals.copy()
    min_residuals[min_residuals > 0] = 0
    dev_costs = - (lambda_up * max_residuals - lambda_down * min_residuals)
    return dev_costs


# 3. MODEL USED TO PREDICT BUYER'S DATA

def aux_model(Xtrain, Ytrain, Xtest, Ytest, lambda_up, lambda_down, nominal_level):
    Yown = QuantReg(Ytrain, Xtrain).fit(q=nominal_level).predict(Xtest)
    g = gain(Ytest, Yown, lambda_up, lambda_down)
    return g


def model(X, Y, lambda_up, lambda_down, nominal_levels):  # model function (train and evaluate the corresponding gain)
    # X: covariates
    # Y: target
    X = pd.DataFrame(X)
    # by construction, own features are in the last column
    X_own = X.iloc[:, len(X.columns) - 1].copy()
    Y = pd.DataFrame(Y)
    # variables of the owner
    Xown_train = pd.DataFrame(X_own.iloc[0:(X.shape[0] - hours_ - 1)])
    Xown_test = pd.DataFrame(X_own.iloc[(X.shape[0] - hours_):])
    # variables from the market
    Xtrain = X.iloc[0:(X.shape[0] - hours_ - 1), :]
    Xtest = X.iloc[(X.shape[0] - hours_):, :]
    Ytrain = Y.iloc[0:(X.shape[0] - hours_ - 1), :]
    Ytest = Y.iloc[(X.shape[0] - hours_):, :]
    nominal_levels_test = nominal_levels.iloc[(X.shape[0] - hours_):]
    lambda_up_test = lambda_up.iloc[(X.shape[0] - hours_):]
    lambda_down_test = lambda_down.iloc[(X.shape[0] - hours_):]
    # electricity market's gain
    g_own = np.repeat(0.0, hours_)
    for level_ in nominal_levels_test.unique():
        h_indexes = (nominal_levels_test == level_)
        y_own = QuantReg(Ytrain, Xown_train).fit(q=level_).predict(Xown_test.values[h_indexes, :])
        g_own[h_indexes] = gain(Ytest.values[h_indexes, 0], y_own, lambda_up_test.values[h_indexes],
                                lambda_down_test.values[h_indexes])
    g_market = np.repeat(0.0, hours_)
    for level_ in nominal_levels_test.unique():
        h_indexes = (nominal_levels_test == level_)
        y_market = QuantReg(Ytrain, Xtrain).fit(q=level_).predict(Xtest.values[h_indexes, :])
        g_market[h_indexes] = gain(Ytest.values[h_indexes, 0], y_market, lambda_up_test.values[h_indexes],
                                   lambda_down_test.values[h_indexes])
    g = g_market - g_own
    return max(0, g.mean())


# 4. DATA ALLOCATION - PAPER'S EQUATION (18)

def data_allocation(p, b, X, noise):
    # Function which receives the current price (p) and bid (b) and decide
    # the quality at this buyer gets allocate
    Xnoise = X + max(0, p - b) * noise
    Xnoise = pd.DataFrame(Xnoise)
    X = pd.DataFrame(X)
    # the last variable is known by the owner!
    Xnoise.iloc[:, X.shape[1] - 1] = X.iloc[:, X.shape[1] - 1]
    return Xnoise


# 5. REVENUE - PAPER'S EQUATION (19)

def revenue(p, b, Y, X, Bmin, epsilon, lambda_up, lambda_down, nominal_levels):
    # Function that computes the final value to be paid by buyer
    reps = 5
    expected_revenue = np.repeat(0.0, reps)
    sigma = X.std().mean()
    for i in range(reps):
        np.random.seed(i)
        noise = np.random.normal(0, sigma, X.shape)

        def f(z):
            XX = data_allocation(p, z, X, noise)
            return model(XX, Y, lambda_up, lambda_down, nominal_levels)

        X_alloc = data_allocation(p, b, X, noise)
        xaxis = np.arange(Bmin, b + epsilon / 2, epsilon)
        if len(xaxis) == 1:
            expected_revenue[i] = max(0, b * model(X_alloc, Y, lambda_up, lambda_down, nominal_levels))
        else:
            I_ = sum([f(v) for v in xaxis]) * (xaxis[1] - xaxis[0])
            expected_revenue[i] = max(0, b * model(X_alloc, Y, lambda_up, lambda_down, nominal_levels) - I_)
    return expected_revenue.mean()


def revenue_posAlloc(p, b, Y, X, noise, Bmin, epsilon, lambda_up, lambda_down, nominal_levels):
    # same as revenue but using fixed noise matrix
    def f(z):
        XX = data_allocation(p, z, X, noise)
        return model(XX, Y, lambda_up, lambda_down, nominal_levels)

    X_alloc = data_allocation(p, b, X, noise)
    xaxis = np.arange(Bmin, b + epsilon / 2, epsilon)
    if len(xaxis) == 1:
        expected_revenue = max(0, b * model(X_alloc, Y, lambda_up, lambda_down, nominal_levels))
    else:
        I_ = sum([f(v) for v in xaxis]) * (xaxis[1] - xaxis[0])
        expected_revenue = max(0, b * model(X_alloc, Y, lambda_up, lambda_down, nominal_levels) - I_)

    return expected_revenue


# 6. PRICE UPDATE - PAPER'S ALGORITHM 2

def aux_price(c_, w_last, b, Y, X, Bmax, delta, Bmin, epsilon, lambda_up, lambda_down, nominal_levels):
    g = revenue(c_, b, Y, X, Bmin, epsilon, lambda_up, lambda_down, nominal_levels)
    w = w_last * (1 - delta) + delta * g
    return w


def price_update(b, Y, X, Bmin, Bmax, epsilon, delta, N, w, lambda_up, lambda_down, nominal_levels):
    c = np.arange(Bmin, Bmax + epsilon / 2, epsilon)
    Wn = np.sum(w)
    probs = w / Wn
    pool = multiprocessing.Pool()
    res = []
    for j, c_ in enumerate(c):
        res.append(pool.apply_async(aux_price, (
        c_, w[j], b, Y, X, Bmax, delta, Bmin, epsilon, lambda_up, lambda_down, nominal_levels)))
    w = np.array([r.get() for r in res])
    w = w.transpose()
    pool.close()
    pool.join()
    # print('weights', w/np.sum(w))
    Wn = np.sum(w)  # this line was missing
    probs = w / Wn
    return probs, w


# 7. PAYMENT DIVISION - PAPER'S ALGORITHM 1

def aux_shap_aprox(m, M, K, X, Y, lambda_up, lambda_down, nominal_levels):
    phi_ = 0
    for k in np.arange(0, K):
        np.random.seed(k)
        sg = np.random.permutation(M)
        i = 1
        while sg[0] == m:
            np.random.seed(i)
            sg = np.random.permutation(M)
            i += 1
        pos = sg[np.arange(0, np.where(sg == m)[0][0])]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        G = model(XX, Y, lambda_up, lambda_down, nominal_levels)
        pos = sg[np.arange(0, np.where(sg == m)[0][0] + 1)]
        XX = X.iloc[:, pos].copy()
        XX[X.columns[M]] = X[X.columns[M]]  # include own seller data
        Gplus = model(XX, Y, lambda_up, lambda_down, nominal_levels)
        phi_ += max(0, Gplus - G)
    return phi_


def shapley_aprox(Y, X, K, lambda_up, lambda_down, nominal_levels):
    M = X.shape[1] - 1
    phi = np.repeat(0.0, M)
    pool = multiprocessing.Pool()
    res = []
    for m in np.arange(0, M):
        res.append(pool.apply_async(aux_shap_aprox, (m, M, K, X, Y, lambda_up, lambda_down, nominal_levels)))
    phi = np.array([r.get() for r in res])
    phi = phi.transpose()
    pool.close()
    pool.join()
    return phi / K


def square_rooted(x):
    return np.round(np.sqrt(sum([a * a for a in x])), 3)


def cos_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return np.round(np.abs(numerator) / np.float(denominator), 3)


def shapley_robust(Y, X, K, lambd, lambda_up, lambda_down, nominal_levels):
    M = X.shape[1] - 1
    phi_ = np.repeat(0.0, M)
    phi = shapley_aprox(Y, X, K, lambda_up, lambda_down, nominal_levels)
    for m in np.arange(0, M):
        s = 0
        for k in np.arange(0, M):
            if k != m:
                s += cos_similarity(X.iloc[:, m], X.iloc[:, k])
        phi_[m] = phi[m] * np.exp(-lambd * s)
    if phi.sum() > 0:
        phi = phi_ / phi_.sum()
    return phi


#
def nominal_level_forecast(lambda_up, lambda_down): # expression (5) of the paper
    T_times = lambda_up.shape[0]

    # construct probabilities
    prob_up = lambda_up - lambda_down
    prob_up[prob_up > 0] = 1
    prob_up[prob_up == 0] = 0.5
    prob_up[prob_up < 0] = 0

    lambda_up_train = lambda_up.iloc[0:(T_times - hours_ - 1)]

    lambda_down_train = lambda_down.iloc[0:(T_times - hours_ - 1)]

    prob_up_train = prob_up.iloc[0:(T_times - hours_ - 1)]

    # train the model with training hours
    mod_lambda_up = SimpleExpSmoothing(lambda_up_train).fit()
    mod_prob_up = SimpleExpSmoothing(prob_up_train).fit()
    mod_lambda_down = SimpleExpSmoothing(lambda_down_train).fit()

    # predict historical hours (train+validation)
    mod_lambda_up_opt = SimpleExpSmoothing(lambda_up).fit(mod_lambda_up.params['smoothing_level'])
    mod_prob_up_opt = SimpleExpSmoothing(prob_up).fit(mod_prob_up.params['smoothing_level'])
    mod_lambda_down_opt = SimpleExpSmoothing(lambda_down).fit(mod_lambda_down.params['smoothing_level'])

    pred_lambda_up = mod_lambda_up_opt.fittedvalues
    pred_prob_up = mod_prob_up_opt.fittedvalues
    pred_lambda_down = mod_lambda_down_opt.fittedvalues

    pred_lambda_up_ = pred_lambda_up * pred_prob_up
    pred_lambda_down_ = pred_lambda_down * (1 - pred_prob_up)

    pred_nominal_levels = pd.DataFrame((20*(pred_lambda_down_ / (pred_lambda_down_ + pred_lambda_up_))).round())/20

    pred_nominal_levels[pred_lambda_down_ == 0] = 0.05
    pred_nominal_levels[pred_nominal_levels == 0] = 0.05
    pred_nominal_levels[pred_nominal_levels == 1] = 0.95
    # todo - Pred nominal levels -> normalizar para intervalos de 0.05
    # sugestao: np.unique(np.round(x * 20) / 20)

    # predict future hour
    n_data = lambda_down.index[lambda_down.shape[0] - 1] + 1
    pred_lambda_up = mod_lambda_up_opt.predict(start=n_data, end=n_data)
    pred_prob_up = mod_prob_up_opt.predict(start=n_data, end=n_data)
    pred_lambda_down = mod_lambda_down_opt.predict(start=n_data, end=n_data)

    pred_lambda_up_ = pred_lambda_up * pred_prob_up
    pred_lambda_down_ = pred_lambda_down * (1 - pred_prob_up)

    pred_nominal_levels_future = pd.DataFrame((pred_lambda_down_ / (pred_lambda_down_ + pred_lambda_up_)).round(2))
    pred_nominal_levels_future[pred_lambda_down_ == 0] = 0.05
    pred_nominal_levels_future[pred_nominal_levels_future == 0] = 0.05
    pred_nominal_levels_future[pred_nominal_levels_future == 1] = 0.95

    return {'pred_nominal_levels_past': pred_nominal_levels, 'pred_nominal_levels_future': pred_nominal_levels_future}
