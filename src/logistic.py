# %%
import numpy as np
import pandas as pd
import math
from pre_process import pre_process
# %%
class LogisticRegression:
    def __init__(self,num_iters = 2000, alpha  = 0.01):
        self.NUM_ITERS = num_iters
        self.ALPHA = alpha
        self.LAMBDA = 0

    def log(self, x):
        g = lambda x: math.log(x)
        return np.vectorize(g)(x)

    def sigmoid(self,z):
        g = lambda z : 1 / (1 + math.exp(-z))
        return np.vectorize(g)(z)

    def costFunction(self,theta, X, Y):
        """ Compute cost and gradient for logistic regression""" 
        m = len(Y)
        h = self.sigmoid(X@theta)
        J = 1 / m  * (- Y.T @ self.log(h) - (1-Y).T @ self.log(1-h))  
        # grad = 1/ m * X.T @ (h - y)
        return J

    def gradientDescent(self,X, y, LAMBDA = None):
        """ Return theta and J_history """ 
        #  number of instances
        if LAMBDA is None:
            LAMBDA = self.LAMBDA
        theta = np.random.rand(X.shape[1],1)
        m = len(y)
        J_history = np.zeros((self.NUM_ITERS,1))
        for i in range(self.NUM_ITERS):
            h = self.sigmoid(X@theta)
            grad = 1 / m * X.T @ (h - y) + LAMBDA * theta
            theta = theta - self.ALPHA * grad 
            J_history[i] = self.costFunction(theta, X, y)
            print(f"Lambda {LAMBDA}, Iter {i} : Cost {J_history[i]}")
        
            
        return theta, J_history
    

    def fit(self,X, y, LAMBDA = None):
        ''' y is column vector ''' 
        if LAMBDA is None:
            LAMBDA = self.LAMBDA
        X, y = np.atleast_2d(X), np.atleast_2d(y)
        self.theta, J_history = self.gradientDescent(X,y, LAMBDA)
        return self.theta

    
    def predict(self, X_test, theta = None):
        if theta is None:
            theta = self.theta
        Y_predict = np.zeros((len(X_test), 1))
        for i in range(len(X_test)):
            if self.sigmoid(X_test[i]@theta) >= 0.5:
                Y_predict[i] = 1
        return Y_predict
    
    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            divisible = len(row_ids) - len(row_ids) % num_folds
            valid_ids = np.split(row_ids[:divisible], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[divisible:])
            train_ids = [[j for j in row_ids if j not in valid_ids[valid_id]] for valid_id in range(num_folds)]
            total_Cost = 0
            for i in range(num_folds):
                valid_part = {
                    'X': X_train[valid_ids[i]],
                    'Y': Y_train[valid_ids[i]]
                }
                train_part = {
                    'X': X_train[train_ids[i]],
                    'Y': Y_train[train_ids[i]]
                }
                theta = self.fit(train_part['X'], train_part['Y'], LAMBDA = LAMBDA)
                # Y_pred = self.predict(valid_part['X'], theta= theta)
                total_Cost += self.costFunction(theta = theta, X = valid_part['X'], Y = valid_part['Y'])
            return total_Cost / num_folds
        
        def range_scan(best_LAMBDA, minium_cost, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_cost = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_cost < minium_cost:
                    best_LAMBDA = current_LAMBDA
                    minium_cost = aver_cost
            return best_LAMBDA, minium_cost

        MAX_LAMBDA = 20
        INIT_COST = 10 ** 10
        INIT_LAMBDA = 0
        best_LAMBDA, minium_cost = range_scan(
            best_LAMBDA = INIT_LAMBDA, minium_cost= INIT_COST, LAMBDA_values= range(MAX_LAMBDA)
        )

        STEP_SIZE = 10
        LAMBDA_values = [k * 1. / STEP_SIZE for k in range(
            max(0, (best_LAMBDA - 1) * STEP_SIZE), (best_LAMBDA + 1) * STEP_SIZE, 1)]
        best_LAMBDA, minium_cost = range_scan(
            best_LAMBDA = best_LAMBDA, minium_cost= minium_cost, LAMBDA_values= LAMBDA_values
        )
        self.LAMBDA = best_LAMBDA
        return best_LAMBDA

# %%
    
if __name__ == '__main__':
    # %%
    X_train, X_test, Y_train, Y_test = pre_process()
    X_train, X_test, Y_train, Y_test = np.atleast_2d(X_train), np.atleast_2d(X_test), np.atleast_2d(Y_train), np.atleast_2d(Y_test)
    Y_train, Y_test = Y_train.T, Y_test.T
    # %%
    logistic = LogisticRegression(num_iters = 2000)
    logistic.get_the_best_LAMBDA(X_train, Y_train)
    logistic.fit(X_train, Y_train)
    pd.DataFrame(X_test)
    # %%
    Y_predict = logistic.predict(X_test)
    print((Y_predict == Y_test).sum() / len(Y_test))
    print(logistic.theta)
    # %%
    Y_train_predict = logistic.predict(X_train)
    print((Y_train_predict == Y_train).sum() / len(Y_train))

    
    # print("du doan", logistic.predict(np.atleast_2d([75,3,0,0,0,1,0,0,0,0,1])))

