# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:27:00 2020

@author: Abdul Qayyum
"""


import torch
from torch import tensor
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline  
path='C:\\Users\\moona\\Desktop\\Mylecture\\LAB7\\LAB7\\data\\Ecommerce Customers'
df = pd.read_csv(path)
df.head(3)
# we want to predict 'Yearly Amount Spent'

t_type = torch.float64

def add_ones(X):
    """
    Add a column of ones at the left hand side of matrix X
    X: (N, d) tensor
    Returns
        (N, d+1) tensor
    """
    ones = torch.ones((X.shape[0],1), dtype=t_type)
    X = torch.cat((ones, X), dim=-1)
    return X

def make_tensor(*args):
    """
    Check if arguments are tensor, converts arguments to tensor
    accepts and returns Iterables
    """
    tensors = [el if torch.is_tensor(el) else tensor(el, dtype=t_type) for el in args ]
    return tensors[0] if len(tensors)==1 else tensors

y = df['Yearly Amount Spent'].values
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']].values
X, y = make_tensor(X,y)

X_b = add_ones(X)
y = y.reshape(-1,1)

def train_test_split(X, y, test_size=0.2):
    N = len(y)
    split_index = int(N*test_size)
    
    indexes = list(range(N))
    np.random.shuffle(indexes)
    
    X_train, X_test = X[split_index:, :], X[:split_index, :]
    y_train, y_test = y[split_index:], y[:split_index]
    
    return X_train, X_test, y_train, y_test

def optimal_weight(X,y):
    assert X.shape[0] == y.shape[0], f"dimensions {X.shape} and {y.shape} does not fit"
    X, y = make_tensor(X, y)
    w_opt = torch.inverse(X.t()@X) @ X.t() @ y
    return w_opt


X_train, X_test, y_train, y_test = train_test_split(X_b, y)

w_opt = optimal_weight(X_train,y_train)
y_pred = X_test @ w_opt

print(w_opt)

def sse(*args):
    return ((args[0]-args[1])**2).mean()

sse(y_pred, y_test)

################################################### Randomly select the dataset#############################
X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)
X, y = make_tensor(X,y)

X_b = add_ones(X)

plt.plot(X.tolist(),y.tolist(),'b.')
w_opt = optimal_weight(X_b,y)
print(w_opt)

xs = X_b[:, 1]
ys = X_b@w_opt
plt.plot(xs.tolist(), ys.tolist(), 'g')

xs1 = torch.linspace(0,2, dtype=t_type)
ys1 = add_ones(xs1.reshape(-1,1))@w_opt
plt.plot(xs1.tolist(), ys1.tolist(), 'r')


def calc_cost(X, y, theta):
    y_pred = X@theta
    return ((y_pred - y)**2).sum()/N   

def calc_gradient(X, y, theta):
    gradient = X.t() @ (X@theta - y)
    return gradient
    
def update_theta(X,y,theta, alpha):
    gradient = calc_gradient(X, y, theta)
    theta_new = (theta - alpha*(gradient)/N)    
    return theta_new

def batch_gradient_descent(X,y,theta,alpha=0.1,max_iter=200, time_alpha=True):
    X, y = make_tensor(X, y)  
    assert X.shape[0] == y.shape[0], "Dimensions must fit"
    
    N, d = X.shape
    theta_history = []
    cost_history = []
    for _ in range(max_iter):
        
        # make alpha time dependent
        if time_alpha:
            alpha_ratio = (max_iter-_/2)/max_iter
            alpha = alpha * alpha_ratio

        theta = update_theta(X,y,theta, alpha)
        cost = calc_cost(X,y,theta)
        
        theta_history.append(theta)
        cost_history.append(cost)
        
    return theta, cost_history, theta_history


def plot_cost_theta(cost_history, theta_history):
    d = len(theta_history[0])

    fig, ax1 = plt.subplots(figsize=(10,6))
    for i in range(d):
        theta_list = [theta[i][0] for theta in theta_history]
        ax1.plot(theta_list, label=f"theta {i}")

    ax1.set_ylabel('Theta', color='r')
    ax1.tick_params('y', colors='r')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(cost_history, 'b')
    ax2.set_ylabel('Cost', color='b')
    ax2.tick_params('y', colors='b')

    plt.show()
    
N, d = X_b.shape
theta_random = torch.randn((d,1), dtype=torch.double)
theta, cost_history, theta_history = batch_gradient_descent(X_b, y, theta_random, alpha=0.1)
plot_cost_theta(cost_history, theta_history)
print(theta)

theta, cost_history, theta_history = batch_gradient_descent(X_b, y, theta_random, alpha=1)
plot_cost_theta(cost_history, theta_history)
print(theta)


def stochastic_gradient_descent(X,y,theta,alpha=0.01,max_iter=20, time_alpha=True, batch_size=20):
    X, y = make_tensor(X, y)  
    assert X.shape[0] == y.shape[0], "Dimensions must fit"
    
    N, d = X.shape
    theta_history = []
    cost_history = []
    for _ in range(max_iter):
        # make alpha time dependent
        if time_alpha:
            alpha_ratio = (max_iter-_/2)/max_iter
            alpha = alpha * alpha_ratio
        
        indices = np.random.permutation(N)
        X, y = X[indices], y[indices]
        
        for i in range(0, N, batch_size):
            X_i, y_i = X[i:i+batch_size], y[i:i+batch_size]
            theta = update_theta(X_i, y_i, theta, alpha)
            cost = calc_cost(X_i, y_i, theta)
            
            theta_history.append(theta)
            cost_history.append(cost)
        
    return theta, cost_history, theta_history


theta, cost_history, theta_history = stochastic_gradient_descent(X_b, y, theta_random, alpha=0.1)
plot_cost_theta(cost_history, theta_history)
print(theta)
########################## varify using the scikit learn function
from sklearn import linear_model
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-2)
clf.fit(X_b, y.flatten())
clf.coef_