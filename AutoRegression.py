import numpy as np
from linear_model import LeastSquares


class My_AutoReg:
    def __init__(self,lag):
        self.lag = lag

    def fit(self,X):
        self.X = self.BuildX(X,self.lag)
        self.y = self.Buildy(X,self.lag)

        model = LeastSquares()
        #print(self.X.shape)
        #print(self.y.shape)
        model.fit(self.X,self.y)
        self.my_model = model



    def BuildX(self,X,lag):
        if type(self.lag) == type(1):
            lag = np.arange(1,self.lag+1)

        length = len(X)
        A = np.empty((0,len(lag)+1),dtype=np.int)
        for i in range(length-lag[-1]):
            xi = np.ones(len(lag)+1,dtype=np.int)

            for j,term in enumerate(lag):
                xi[j+1] = X[i+term-1]
            A = np.vstack([A, xi])
        #print(A)

        return A

    def Buildy(self,X,lag):
        if type(self.lag) == type(1):
            lag = np.arange(1,self.lag+1)

        maxlag = lag[-1]
        y=np.empty(int(0),dtype=np.int)
        length = len(X)
        for i in range(length-lag[-1]):
            y = np.append(y, X[i+lag[-1]])
        #print(y)
        return y

    def predict(self, Z, start = None, end = None):
        n_time_topre = end - start + 1
        start = start - self.lag
        model = self.my_model
        w = model.w
        y_pred = [0]*n_time_topre
        latest_X = self.X[start-1]
        y_pred[0] = np.dot(latest_X,w)
        i=1
        while(i<n_time_topre):
            if(start + i >= len(Z)):
                latest_X = np.roll(latest_X,-1)
                latest_X[0] = 1
                latest_X[-1] = y_pred[i-1]
                y_pred[i] = model.predict(latest_X)
                i += 1
            else:
                latest_X = self.X[start+i-1]
                y_pred[i] = model.predict(latest_X)
                i += 1
        return y_pred

class AutoReg1():
    def __init__(self,lags):
        self.lags = lags

    def fit(self,y):
        self.y = np.zeros(len(y)) #train set
        self.y = y
        self.X = self.BuildX() #matrix constructed from train set by shifting
        model = LeastSquares()
        #print(self.X.shape)
        #print(self.y.shape)
        model.fit(self.X,self.y[self.lags:].T)
        self.w = model.w
        #print(self.w)
        # self.my_model = model

    def BuildX(self):
        k = self.lags
        y = self.y
        T = len(y)
        X = np.ones((T-k,k+1))
        for t in range(k,T):
            X[t-k][1:] = y[t-k:t]
        #print(X)
        #print(y)
        return X
     
    def predict(self,start,end):
        y=self.y
        k=self.lags
        T= len(y)
        y_pred = np.zeros(end-start+1)

        if start <= T:
            ylag = y[start-k:start]
        else:
            ylag = y[T-k:]
            #predict from y[T] for each y[t] up to y[start-1]
            for t in range(T,start):    
                y_pred1 = ylag@self.w[1:]+self.w[0] 
                ylag = np.roll(ylag,-1)
                ylag[-1] =y_pred1
        #scroll data into y_pred 
        for t in range(start,end+1):
            y_pred[t-start] = ylag@self.w[1:]+self.w[0]
            ylag = np.roll(ylag,-1)
            ylag[-1] =y_pred[t-start]
        return y_pred            
         

