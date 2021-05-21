import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        ''' YOUR CODE HERE '''
        n, d = X.shape
        V = np.diag(z)
        self.w = solve(X.T@V@X, X.T@V@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.T[0], lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        n,d = X.shape
        ''' MODIFY THIS CODE '''
        # Calculate the function value
        #f = 0.5*np.sum((X@w - y)**2)
        f=0
        g=np.zeros((d,1))
        for i in range(n):
            ri = w.T@X[i]-y[i]
            f += np.log(2*np.cosh(ri))
            g += X[i]*np.tanh(ri)
        # Calculate the gradient value
        #g = X.T@(X@w-y)
            
            
        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        n, d = X.shape
        Z=np.hstack((np.ones((n,1)),X))
        
        self.v = solve(Z.T@Z, Z.T@y)
        self.w = self.v[1:]

    def predict(self, X):
        n, d = X.shape
        
        return X@self.w + np.ones((n,1))*self.v[0]

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):

        Z=self.__polyBasis(X)
        self.v = solve(Z.T@Z, Z.T@y)
        #self.w = self.v[1:]

    def predict(self, X):

        Z=self.__polyBasis(X)
        return Z@self.v

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        n, d = X.shape
        Z=np.ones((n,1))
        if self.p !=0:
            for k in range(1,self.p+1):
                Z=np.hstack((Z,X**k))
        return Z
        
