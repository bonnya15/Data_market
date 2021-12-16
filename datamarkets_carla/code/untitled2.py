# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:18:08 2021

@author: shiuli Subhra Ghosh
"""

from math import sqrt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
#from sklearn.linear_model import QuantileRegressor


dfX = pd.read_csv('X_VAR3.csv')
dfY = pd.read_csv('Y_VAR3.csv')
Data_X=dfX.iloc[:,1]
Data_Y=dfY.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(Data_X, Data_Y, test_size=0.05, random_state=0)
X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)



preds2=pd.DataFrame()
preds2['X_test']=list(X_test)
preds2['y_test']=list(y_test)
for tau in [0.1,0.3,0.5,0.7,0.9]:
    clf = LGBMRegressor(objective='quantile', alpha=tau)
    clf.fit(X_train, y_train)
    preds2[str(tau)] = list(clf.predict(X_test))
    
    

plt.figure(figsize=(15, 8))
plt.scatter(X_test,y_test,label="X_test")
plt.scatter(X_test,preds2['0.1'],label='0.1')
plt.scatter(X_test,preds2['0.3'],label='0.3')
plt.scatter(X_test,preds2['0.5'],label='0.5')
plt.scatter(X_test,preds2['0.7'],label='0.7')
plt.scatter(X_test,preds2['0.9'],label='0.9')
plt.legend()
plt.show()    



import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


data=pd.DataFrame()
data=pd.concat([X_train,y_train.rename("y")],axis=1)

data['x']=Data_X
data['y']=Data_Y

mod = smf.quantreg('y ~ x',data)
res = mod.fit(q=0.5)
for i in res.params.index:
    print(res.params[i])
    
    
quantiles = np.arange(0.05, 0.96, 0.1)


def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params["Intercept"], res.params["x"]] + res.conf_int().loc[
        "x"
    ].tolist()


models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])

ols = smf.ols("y ~ x", data).fit()
ols_ci = ols.conf_int().loc["x"].tolist()
ols = dict(
    a=ols.params["Intercept"], b=ols.params["x"], lb=ols_ci[0], ub=ols_ci[1]
)

print(models)
print(ols)  

x = np.arange(data.x.min(), data.x.max(),0.01)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle="dotted", color="red")

y = get_y(ols["a"], ols["b"])

ax.plot(x, y, color="red", label="OLS")
ax.scatter(data.x, data.y, alpha=0.2)
ax.set_xlim((-8, 8))
ax.set_ylim((-8, 8))
legend = ax.legend()
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)  