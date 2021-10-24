import pandas as pd
import numpy as np
data = pd.read_csv("dataset.csv")
Y = data["Profit"]
del data["Profit"]
bias = pd.Series(1,index=range(len(Y)))
data["Bias"] = bias
Header_X_Bias = list(data.columns.values)
Header_X_Bias = Header_X_Bias[:-1]
Header_X_Bias.insert(0,"Bias")
data = data[Header_X_Bias]
X = np.array(data)
Y = np.array(Y)
Y = Y.reshape(len(Y),1)
Theta = [0,0]
Theta = np.array(Theta)
Theta = Theta.reshape(2,1)
alpha = 0.01
Iterations = 1500

def cost(X, Y, Theta):
    Hypothesis = np.dot(X, Theta)
    Error = Y - Hypothesis
    # Matrix method for calculating Cost
    Cost = np.dot(Error.T, Error) / (2 * len(Error))
    return Cost[0][0]


def gradient(X, Y, Theta, Iterations, alpha):
    for i in range(Iterations):
        Loss = Y - np.dot(X, Theta) + (np.dot(Theta.T, Theta) * 0.001)
        Cost = cost(X, Y, Theta)
        Loss = Loss * (-1)
        dJ = (np.dot(X.T, Loss) * 2) / len(Y)
        Theta = Theta - (alpha * dJ)
    return Theta
Theta = gradient(X,Y,Theta, Iterations, alpha)


def online_train(Xn, Yn, Theta):
    Loss = Yn - np.dot(Xn, Theta) + (np.dot(Theta.T, Theta) * 0.001)
    Loss = Loss * (-1)
    dJ = (np.dot(Xn.T, Loss) * 2) / len(Y)
    Theta = Theta - (alpha * dJ)
    return Theta

#Theta = online_train(X_new, Y_new, Theta)