import numpy as np
import math
from pre_process import pre_process

def log(x):
    g = lambda x: math.log(x)
    return np.vectorize(g)(x)

def sigmoid(z):
    g = lambda z : 1 / (1 + math.exp(-z))
    return np.vectorize(g)(z)

def costFunction(theta, X, y):
    """ Compute cost and gradient for logistic regression""" 
    m = len(y)
    h = sigmoid(X@theta)
    J = 1 / m  * (- y.T @ log(h) - (1-y).T @ log(1-h))  
    # grad = 1/ m * X.T @ (h - y)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """ Return theta and J_history """ 
    #  number of instances
    m = len(y)
    J_history = np.zeros((num_iters,1))
    for i in range(num_iters):
        h = sigmoid(X@theta)
        grad = 1 / m * X.T @ (h - y)
        theta = theta - alpha * grad 
        J_history[i] = costFunction(theta, X, y)
        
    return theta, J_history

class LogisticRegression:
    def __init__(self,num_iters = 2000, alpha  = 0.01):
        self.NUM_ITERS = num_iters
        self.ALPHA = alpha

    def log(self, x):
        g = lambda x: math.log(x)
        return np.vectorize(g)(x)

    def sigmoid(self,z):
        g = lambda z : 1 / (1 + math.exp(-z))
        return np.vectorize(g)(z)

    def costFunction(self,theta, X, y):
        """ Compute cost and gradient for logistic regression""" 
        m = len(y)
        h = self.sigmoid(X@theta)
        J = 1 / m  * (- y.T @ self.log(h) - (1-y).T @ self.log(1-h))  
        # grad = 1/ m * X.T @ (h - y)
        return J

    def gradientDescent(self,X, y, theta):
        """ Return theta and J_history """ 
        #  number of instances
        m = len(y)
        J_history = np.zeros((self.NUM_ITERS,1))
        for i in range(self.NUM_ITERS):
            h = sigmoid(X@theta)
            grad = 1 / m * X.T @ (h - y)
            theta = theta - self.ALPHA * grad 
            J_history[i] = self.costFunction(theta, X, y)
        
            
        return theta, J_history
    

    def fit(self,X, y):
        ''' y is column vector ''' 
        X, y = np.atleast_2d(X), np.atleast_2d(y)
        self.m, self.n = np.shape(X)
        X = np.concatenate((np.ones([self.m,1]), X), axis= 1)
        self.theta = np.zeros((self.n+1, 1))
        self.theta, J_history = self.gradientDescent(X,y, self.theta)

    
    def predict(self, X_test):
        self.m_test, self.n_test = np.shape(X_test)
        X_test = np.concatenate((np.ones([self.m_test,1]), X_test), axis= 1)

        Y_predict = np.zeros((len(X_test), 1))
        for i in range(len(X_test)):
            if sigmoid(X_test[i]@self.theta) >= 0.5:
                Y_predict[i] = 1
                
        return Y_predict


    
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = pre_process()
    print(X_test)
    print(X_train)
    X_train, X_test, Y_train, Y_test = np.atleast_2d(X_train), np.atleast_2d(X_test), np.atleast_2d(Y_train), np.atleast_2d(Y_test)
    Y_train, Y_test = Y_train.T, Y_test.T

    logistic = LogisticRegression(num_iters = 3000)
    logistic.fit(X_train, Y_train)
    print(X_test)
    Y_predict = logistic.predict(X_test)
    print((Y_predict == Y_test).sum() / len(Y_test))
    print(logistic.theta)
    
    print("du doan", logistic.predict(np.atleast_2d([75,3,0,0,0,1,0,0,0,0,1])))

