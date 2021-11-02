import numpy as np
import pandas as pd


class Online_SGD:
    import warnings
    warnings.filterwarnings("ignore")


    def __init__(self,  weight, bias, learning_rate=0.2,damp_factor=1.02):
        self.learning_rate = learning_rate
        self.damp_factor = damp_factor
        self.w = weight
        self.b = bias

    def fit_online(self, X, Y):

        for i in range(X.shape[0]):
            x = (np.array(X))[i].reshape(1, X.shape[1])
            y = (np.array(Y))[i].reshape(1, 1)
            Lw = np.dot((y - np.dot(x, self.w.T) - self.b), x)
            Lb = (y - np.dot(x, self.w.T) - self.b)
            self.w = self.w + self.learning_rate * Lw
            self.b = self.b + self.learning_rate * Lb
            self.learning_rate = self.learning_rate / self.damp_factor

        return self.w, self.b

    def fit_regression(self, X, Y):
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, Y)
        self.w = reg.coef_.reshape(1, X.shape[1])
        self.b = reg.intercept_.reshape(1, 1)
        # a_file = open(self.w_file, "w")
        # for row in self.w:
        #     np.savetxt(a_file, row)
        # a_file.close()
        # b_file = open(self.b_file, "w")
        # for row in self.b:
        #     np.savetxt(b_file, row)
        # b_file.close()

        return self.w, self.b

    def predict(self, X):
        # self.w = np.loadtxt(self.w_file).reshape(1, X.shape[1])  # Reading weights and bias
        # self.b = np.loadtxt(self.b_file).reshape(1, 1)
        m = np.dot(X, self.w.T) + self.b
        n = m.reshape(-1, )

        return n

#     def fit_batch(self,X,Y):
#         self.w = np.zeros((1, X.shape[1]))  # Randomly initializing weights
#         self.b = np.zeros((1, 1))
#         n=1
#         while n<=self.n_epochs:
#             temp = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis=1)
#             temp2=temp.sample(self.k)
#             X_tr = temp2.iloc[:,0:-1]
#             Y_tr = temp2.iloc[:,-1]
# #             print("shape of x_tr",X_tr.shape)
# #             print("temp shape",temp.shape,temp2.shape)
#             Lw=np.zeros((1, X_tr.shape[1]))
#             Lb=np.zeros((1, 1))

#             for i in range(self.k):
#                 x=(np.array(X_tr))[i].reshape(1,X_tr.shape[1])
#                 y=(np.array(Y_tr))[i].reshape(1,1)
# #                 print("shape of x",x.shape)
# #                 print("shape of y",y.shape)
#                 Lw = Lw+(np.dot((y - np.dot(x, self.w.T) - self.b),x))
#                 Lb = Lb+((y - np.dot(x, self.w.T) - self.b))
#             self.w = self.w + self.learning_rate * Lw
#             self.b = self.b + self.learning_rate * Lb
#         #             print("W",self.w)
#         #             print("b",self.b)


#             self.learning_rate = self.learning_rate / self.damp_factor
#             n=n+1
#         return self.w,self.b


