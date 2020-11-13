# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:51:43 2020

@author: Abdul Qayyum
"""

#%% Logisitc Regression 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import tensor
#from utils import add_ones, make_tensor, minmax_scale, t_type

#%matplotlib inline

# basic function
import torch

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
    tensors = [el if torch.is_tensor(el) else torch.tensor(el, dtype=t_type) for el in args ]
    return tensors[0] if len(tensors)==1 else tensors

def minmax_scale(X):
    """
    X: 2 dim. numpy array or torch tensor
    """
    N, d = X.shape
    for i in range(d):
        col = X[:, i]
        col_max, col_min = col.max(), col.min()
        if col_max == col_min:
            continue
        else:
            X[:, i] = (col - col_min) / (col_max - col_min)
    return X
######################################## load dataset ###########################################
df = sns.load_dataset("iris")
df["class"] = df.species.apply(lambda x: 1 if x=='setosa' else 0)
df.head()


X = df[df.columns[:3]].values
y = df["class"].values
y = y.reshape(-1,1)

################################################ define function ###################################
def sigmoid(z):
    # z: torch.float64
    return 1/(1+torch.exp(-z))

def gradient(X, y, theta):
    z = X@theta
    return X.t()@(y - sigmoid(z))

def log_likelihood(X, y, theta):
    z = X@theta
    return y.t()@torch.log(sigmoid(z)) + (1-y).t()@torch.log(sigmoid(-z))

def logistic_regression_function(X, y, n_iter = 1000, step_size = 0.01):
    X, y = make_tensor(X, y)
    X = minmax_scale(add_ones(X))
    y = y.reshape(-1, 1)
    
    N, d = X.shape
    theta = torch.zeros((d,1), dtype=t_type)
    ll = []
    theta_list = []
    for i in range(n_iter):
        grad = gradient(X, y, theta)
        # update theta via gradient ascent
        # maximise log likelihood
        theta = theta + step_size * grad
        ll.append(log_likelihood(X,y,theta).item())
        theta_list.append(theta)
    return theta, ll, theta_list

theta, ll, theta_list = logistic_regression_function(X, y, n_iter=1000)
plt.plot(ll)

df = sns.load_dataset("iris")
df.head()
y = df.species

k = 'setosa'
y_k = y.apply(lambda x: 1 if x==k else 0)
y_k = make_tensor(y_k.values)


class LogisticRegression:
    def __init__(self, alpha=0.01, max_iter=1000, fit_intercept=True):
        self.alpha = alpha # learning rate
        self.max_iter = max_iter
        self.__fit_intercept = fit_intercept
        self.loss_history = []
        self.theta_history = []
        
    def __sigmoid(self, z):
        # z: torch.float64
        return 1/(1+torch.exp(-z))

    def __gradient(self, X, y, theta):
        z = X@theta
        return X.t()@(y - self.__sigmoid(z))

    def log_likelihood(self, X, y, theta):
        z = X@theta
        return y.t()@torch.log(self.__sigmoid(z)) + (1-y).t()@torch.log(self.__sigmoid(-z))
    
    def fit(self, X,y):
        """
        X: (N, d) matrix (iterable)
        y: (N, 1) column vector (iterable)
        """
        X, y = make_tensor(X, y)
        assert X.shape[0] == y.shape[0], "Dimensions must fit"
        X = minmax_scale(X) # scale
        if self.__fit_intercept:
            X = add_ones(X)
        N, d = X.shape
        
        theta = torch.zeros((d,1), dtype=t_type) # initialize gradient
        # reset history
        self.loss_history.clear()
        self.theta_history.clear()
        for i in range(self.max_iter):
            grad = self.__gradient(X, y, theta)
            # update theta via gradient ascent
            # maximise log likelihood
            theta = theta + self.alpha * grad
            self.loss_history.append(-self.log_likelihood(X,y,theta).item())
            self.theta_history.append(theta)    
        self.theta = theta

lm = LogisticRegression()

y = y.apply(lambda x: 1 if x=='setosa' else 0).values.reshape(-1,1)
lm.fit(X, y)
plt.plot(lm.loss_history)

