# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:56:44 2020

@author: Abdul Qayyum
"""

############################################# Linear Regression model ####################################
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from scipy import stats
from sklearn import datasets
from IPython.display import display, HTML


path='C:\\Users\\moona\\Desktop\\Lecture7(Regression_models)\\LAB7\\data\\auto.csv'
auto_df = pd.read_csv(path)
# Remove missing values
auto_df = auto_df.drop(auto_df[auto_df.values == '?'].index)
auto_df = auto_df.reset_index()
auto_df.head()

# Convert quantitive datatypes to numerics
datatypes = {'quant': ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year'],
             'qual': ['origin', 'name']}

quants = auto_df[datatypes['quant']].astype(np.float_)

auto_df = pd.concat([quants, auto_df[datatypes['qual']]], axis=1) 


################################################ Q1##########################################
# Let's start by implementing linear regression from scratch
# using numpy linear algebra

intercept_const = pd.DataFrame({'intercept': np.ones(auto_df.shape[0])})

X = pd.concat([intercept_const, auto_df['horsepower']], axis=1)
y = auto_df['mpg']

def linear_model(X, y):
    """Estimation of parameteres for linear regressions model
    by Ordinary Least Squares (OLS)"""
    XTX_inv = np.linalg.inv(X.T @ X)
    XTy     = X.T @ y
    beta    = XTX_inv @ XTy
    return beta

def predict(beta, X):
    """Perdict y given beta parameters and X variables"""
    return X @ beta

beta = linear_model(X, y)
y_pred = predict(beta, X)


# Our handrolled linear model doesn't provide the useful statistics 
# that summary(lm()) would provide in R. 
# For each estimated coefficient we would like to get the follow stats
# - coefficient 
# - standard error
# - t-value
# - p-value

# Add constant for bias variable
intercept_const = pd.DataFrame({'intercept': np.ones(auto_df.shape[0])})
#Load data
X = pd.concat([intercept_const, auto_df['horsepower']], axis=1)
y = auto_df['mpg']

# Predict coefficients and responses
coefficients = linear_model(X, y)
y_pred       = predict(beta, X)


# Calculate Mean Squared Error
MSE = np.sum(np.square(y_pred - y)) / y.size

# Variance of each variable in X
variance = MSE * (np.linalg.inv(X.T @ X).diagonal())  # To discuss: how is variance derived from MSE?

# Standard error of each variable in X
# given by Var(X) = SE(X)^2
standard_error = np.sqrt(variance)

# t-statistic given by t = β - 0 / SE(β)
t_statistic = coefficients / standard_error

# p-values
p_values = 2*(1 - stats.t.cdf(X.shape[0], np.abs(t_statistic)))


# Present results
results = pd.DataFrame({'feature': X.columns,
                        'coefficients': coefficients,
                        'standard_error': standard_error,
                        't-statistic': t_statistic,
                        'P>|t|': p_values})

results.set_index('feature')


# The statsmodels library provides a convenient means to get the
# same statistics
X = auto_df['horsepower']
X = sm.add_constant(X)     # add bias constant
y = auto_df['mpg']

results = sm.OLS(y, X).fit()
print(results.summary())


# predicted mpg associated with a horsepower of 98

def predict(model, X):
    return model.T @ X

X_ex = np.array([1, 98])

y_ex = predict(coefficients, X_ex)
print(str(np.round(y_ex, 3)) + ' mpg')

# the associated 95% confidence and prediction intervals

model_min = results.conf_int(alpha=0.05)[0]
model_max = results.conf_int(alpha=0.05)[1]

confidence_interval = [predict(model_min, X_ex), predict(model_max, X_ex)]
print(confidence_interval)

#########################################################(b) #######################################
# Let's plot our predicted regression

df = pd.concat([auto_df['horsepower'], auto_df['mpg']], axis=1)
ax = sns.scatterplot(x='horsepower', y='mpg', data=df)
ax.plot(auto_df['horsepower'], y_pred);

####################################################################(c)#######################
# Functions to emulate R's lm().plot() functionality
# Providing powerful residual plots for simple AND multivariate
# linear regresssion
# - bring your own predictions
# - underlying stats available as pandas dataframe
# - visualise linearity and outliers in multiple dimensions


def lm_stats(X, y, y_pred):
    """ LEVERAGE & STUDENTISED RESIDUALS
    - https://en.wikipedia.org/wiki/Studentized_residual#How_to_studentize
    """
    # Responses as np array vector
    try: 
        y.shape[1] == 1
        # take first dimension as vector
        y = y.iloc[:,0]
    except:
        pass
    y = np.array(y)
    
    # Residuals
    residuals = np.array(y - y_pred)
    
    # Hat matrix
    H = np.array(X @ np.linalg.inv(X.T @ X)) @ X.T
    
    # Leverage
    h_ii = H.diagonal()
    
    ## Externally studentised residual
    # In this case external studentisation is most appropriate 
    # because we are looking for outliers.
    
    # Estimate variance (externalised)
    σi_est = []
    for i in range(X.shape[0]):
        # exclude ith observation from estimation of variance
        external_residuals = np.delete(residuals, i)
        σi_est += [np.sqrt((1 / (X.shape[0] - X.shape[1] - 1)) * np.sum(np.square(external_residuals)))]
    σi_est = np.array(σi_est)
    
    # Externally studentised residuals
    t = residuals / σi_est * np.sqrt(1 - h_ii)
    

    # Return dataframe
    return pd.DataFrame({'residual': residuals,
                         'leverage': h_ii, 
                         'studentised_residual': t,
                         'y_pred': y_pred})


def lm_plot(lm_stats_df): 
    """Provides R style residual plots based on results from lm_stat()"""
    # Parse stats
    t      = lm_stats_df['studentised_residual']
    h_ii   = lm_stats_df['leverage']
    y_pred = lm_stats_df['y_pred']
    
    # setup axis for grid
    plt.figure(1, figsize=(16, 18))
    
    # Studentised residual plot
    plt.subplot(321)
    ax = sns.regplot(x=y_pred, y=t, lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('Studentised residuals')
    plt.title('Externally studentised residual plot', fontweight='bold')
    # Draw Hastie and Tibshirani's bounds for possible outliers
    ax.axhline(y=3, color='r', linestyle='dashed')
    ax.axhline(y=-3, color='r', linestyle='dashed');
    
    # Normal Q-Q plot
    plt.subplot(322)
    ax = stats.probplot(t, dist='norm', plot=plt)
    plt.ylabel('Studentised residuals')
    plt.title('Normal Q-Q', fontweight='bold')
    
    # Standardised residuals
    plt.subplot(323)
    ax = sns.regplot(x=y_pred, y=np.sqrt(np.abs(t)), lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('√Standardized residuals')
    plt.title('Scale-Location', fontweight='bold')
    
    # Residuals vs Leverage plot
    plt.subplot(324)
    ax = sns.scatterplot(x=h_ii, y=t)
    plt.xlabel('Leverage')
    plt.ylabel('Studentised residuals')
    plt.title('Externally studentised residual vs Leverage', fontweight='bold');


X = pd.concat([auto_df['horsepower']], axis=1)
# Create the Design Matrix by adding constant bias variable
intercept_const = pd.DataFrame({'intercept': np.ones(X.shape[0])})
X = np.array(pd.concat([intercept_const, X], axis=1))

y = auto_df['mpg']

lm_plot(lm_stats(X, y, y_pred))

def lm_residual_corr_plot(lm_stats_df):
    r = lm_stats_df['residual']
    # Residuals correlation
    plt.figure(1, figsize=(16, 5))
    ax = sns.lineplot(x=list(range(r.shape[0])), y=r)
    plt.xlabel('Observation')
    plt.ylabel('Residual')
    plt.title('Correlation of error terms', fontweight='bold');  

lm_residual_corr_plot(lm_stats(X, y, y_pred))

#################################################################Q(2)#############################
#(a) Produce a scatterplot matrix which includes all of the variables in the data set.

sns.pairplot(auto_df);
#(b) Compute the matrix of correlations between the variables using the function cor(). 
#You will need to exclude the name variable, which is qualitative.
corr_matrix = auto_df.corr().abs()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=1, square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0);
#############################################################(c)##################################

f = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + C(origin)'

y, X = patsy.dmatrices(f, auto_df, return_type='dataframe')
model = sm.OLS(y, X).fit()
print(model.summary())

# The following predictors have P-values < 0.05 which suggests we can reject 
# the null hypothesis that they have no relationship with the response:
model.pvalues[model.pvalues < 0.05].sort_values()
#%%
path1='C:\\Users\\moona\\Desktop\\Lecture7(Regression_models)\\LAB7\\data\\Carseats.csv'
# Load data
carseats = pd.read_csv(path1).drop('Unnamed: 0', axis=1)

# No missing values found

# Pre-processing
# Convert quantitive datatypes to numerics
datatypes = {'quant': ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education'],
             'qual': ['ShelveLoc', 'Urban', 'US']}
# Use floats for all quantitive values
quants = carseats[datatypes['quant']].astype(np.float_)
carseats_df = pd.concat([quants, carseats[datatypes['qual']]], axis=1)

carseats_df.head()

# Feature engineering
f = 'Sales ~ Price + C(Urban) + C(US)'
y, X = patsy.dmatrices(f, carseats_df, return_type='dataframe')

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Make predictions
y_pred = np.array(model.predict(X))

# Feature engineering
f = 'Sales ~ Price + C(US)'
y, X = patsy.dmatrices(f, carseats_df, return_type='dataframe')

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Make predictions
y_pred = np.array(model.predict(X))

# Extract 95% confidence intervals
conf_inter_95 = model.conf_int(alpha=0.05)
conf_inter_95.rename(index=str, columns={0: "min.", 1: "max.",})
#%%
#######################################################Q(4)########################################
np.random.seed(1)
x  = np.random.normal(size=100)
y  = 2*x + np.random.normal(size=100)
df = pd.DataFrame({'x': x, 'y': y}) 
sns.scatterplot(x=x, y=y, color='b');
###################################################################(a)#############################
# Linear regression without intercept
f = 'y ~ x + 0'
y, X = patsy.dmatrices(f, df, return_type='dataframe')

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary(alpha=0.05))
###################################################################(b)#############################
# Linear regression without intercept
f = 'x ~ y + 0'
y, X = patsy.dmatrices(f, df, return_type='dataframe')

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary(alpha=0.05))

# Fit model
model_a = smf.ols(formula='y ~ x', data=df).fit()
model_b = smf.ols(formula='x ~ y', data=df).fit()

print(model_a.tvalues)
print(model_b.tvalues)

np.random.seed(2)
x = np.random.normal(size=100)
y = 1.2*x
df = pd.DataFrame({'x': x, 'y': y})
sns.scatterplot(x=x, y=y);

model_a = smf.ols(formula='y ~ x', data=df).fit()
model_b = smf.ols(formula='x ~ y', data=df).fit()

print(model_a.params)
print(model_b.params)

x = np.random.normal(size=100)
y = x
df = pd.DataFrame({'x': x, 'y': y})
sns.scatterplot(x=x, y=y);

model_a = smf.ols(formula='y ~ x', data=df).fit()
model_b = smf.ols(formula='x ~ y', data=df).fit()

print(model_a.params)
print(model_b.params)
#%%
#(b) Generate an example in R with n = 100 observations in which the coefficient estimate for the regression of X onto Y is different 
#from the coefficient estimate for the regression of Y onto X.

np.random.seed(1)

mu, sigma = 0, 1
x = np.random.normal(mu, sigma, 100)

mu, sigma = 0, 0.25
eps = np.random.normal(mu, sigma, 100)

y = -1 + 0.5*x + eps

print('y length: ' + str(np.linalg.norm(y)))

ax = sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y');

model = smf.ols(formula='y ~ x', data=pd.DataFrame({'x':x, 'y':y})).fit()
print(model.summary())

y_pred = model.predict()
y_act  = -1+(0.5*x)

plt.figure(figsize=(8,8))
ax = sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
ax.plot(x, model.predict(), color='r')
ax.plot(x, -1+(0.5*x), color='y')
ax.legend(['Least squares', 'Population regression']);

########################################(f)####################################
f = 'y ~ x + np.square(x)'
model = smf.ols(formula=f, data=pd.DataFrame({'x':x, 'y':y})).fit()
print(model.summary())

# predict
y_pred = model.predict()
y_act  = -1+(0.5*x)

# plot
plt.figure(figsize=(8,8))
ax = sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
ax.plot(x, model.predict(), color='r')
ax.plot(x, -1+(0.5*x), color='y')
ax.legend(['Least squares quadratic', 'Population regression']);


####################################repeat steps #######################
np.random.seed(1)

# generate x
mu, sigma = 0, 1
x = np.random.normal(mu, sigma, 100)

# generate epsilon
mu, sigma = 0, 0.05
eps = np.random.normal(mu, sigma, 100)

# generate y
y = -1 + 0.5*x + eps
print('y length: ' + str(np.linalg.norm(y)))

# Fit model
model = smf.ols(formula='y ~ x', data=pd.DataFrame({'x':x, 'y':y})).fit()
print(model.summary())

# Make predictions
y_pred = model.predict()
y_act  = -1+(0.5*x)

# Plot results
plt.figure(figsize=(8,8))
ax = sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
ax.plot(x, model.predict(), color='r')
ax.plot(x, -1+(0.5*x), color='y')
ax.legend(['Least squares', 'Population regression']);


np.random.seed(1)

# generate x
mu, sigma = 0, 1
x = np.random.normal(mu, sigma, 100)

# generate epsilon
mu, sigma = 0, 0.5
eps = np.random.normal(mu, sigma, 100)

# generate y
y = -1 + 0.5*x + eps
print('y length: ' + str(np.linalg.norm(y)))

# Fit model
model = smf.ols(formula='y ~ x', data=pd.DataFrame({'x':x, 'y':y})).fit()
print(model.summary())

# Make predictions
y_pred = model.predict()
y_act  = -1+(0.5*x)

# Plot results
plt.figure(figsize=(8,8))
ax = sns.scatterplot(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
ax.plot(x, model.predict(), color='r')
ax.plot(x, -1+(0.5*x), color='y')
ax.legend(['Least squares', 'Population regression']);


#######################################################################(a)##########################
#write function
np.random.seed(1)
x1 = np.random.uniform(size=100)
x2 = 0.5*x1 + np.random.randn(100)/10
y  = 2 + 2*x1 + 0.3*x2 + np.random.randn(100)
#######################################################################(b)##########################
#(b) What is the correlation between x1 and x2? Create a scatterplot displaying the relationship between the variables.
df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
sns.scatterplot(x='x1', y='x2', data=df);

print('Correlation coefficient: ' + str(np.corrcoef(x1, x2)[0][1]))

##########################################################################(c)###########################

df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'])
ax.view_init(30, 185)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# Fit model
f = 'y ~ x1 + x2'
model = smf.ols(formula=f, data=df).fit()
print(model.summary())


###########################################################(d)##################################################

# Fit model
f = 'y ~ x1'
model = smf.ols(formula=f, data=df).fit()
print(model.summary())

df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
ax = sns.scatterplot(x='x1', y='y', data=df);
plt.xlabel('x1')
plt.ylabel('y')
ax.plot(df['x1'].sort_values(), model.predict(df['x1'].sort_values()), color='r')
ax.plot(x, 2+(2*x), color='y')
ax.legend(['Least squares', 'Population regression']);

#######################################################(e) ##################################################

# Fit model
f = 'y ~ x2'
model = smf.ols(formula=f, data=df).fit()
print(model.summary())


df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
ax = sns.scatterplot(x='x2', y='y', data=df);
plt.xlabel('x2')
plt.ylabel('y')
ax.plot(df['x2'].sort_values(), model.predict(df['x2'].sort_values()), color='r')
ax.plot(x, 2+(0.3*x), color='y')
ax.legend(['Least squares', 'Population regression']);


#df2 = df.append({'x1': 0.1, 'x2': 0.8, 'y': 6}, ignore_index=True)
#
#
## Fit models
#model_c = smf.ols(formula='y ~ x1 + x2', data=df2).fit()
#model_d = smf.ols(formula='y ~ x1', data=df2).fit()
#model_e = smf.ols(formula='y ~ x2', data=df2).fit()
#
#model_c.summary()
#model_d.summary()
#model_e.summary()

