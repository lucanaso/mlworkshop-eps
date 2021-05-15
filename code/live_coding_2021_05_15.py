# Author: Luca Naso
# Creation date: 15 May 2021
# Topic: live coding during Machine Learning Workshop for EPS
#        - a simple example of linear regression with synthetic data
#        - model validation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

np.random.seed(0)
plotting = False

obs_number = 200

# features
X = 1 + 2 * np.random.random(size=(obs_number, 1))

# targets
a = 3.5
b = 8
fluctuations = 1.0
y = b + a * X.squeeze() + fluctuations * np.random.randn(obs_number)

# plot
plt.plot(X, y, '+')
plt.show()

# Simple Linear Regression - SLR
#
# create a SLR model
slr_model = LinearRegression()
# doc: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# train the model
slr_model.fit(X, y)
# Predictions
# - single-point prediction
X_pred = np.array([2])[:, np.newaxis]
# - multiple-point prediction
X_pred = np.array([2, 3])[:, np.newaxis]
first, last, new_obs = X.min(), X.max(), 100
X_pred = np.linspace(first, last, new_obs)[:, np.newaxis]
#
y_pred = slr_model.predict(X=X_pred)
# Plot predictions to chart
if plotting:
    plt.plot(X, y, '+')
    plt.plot(X_pred, y_pred, 'r-')
    plt.show()
#
c0 = slr_model.intercept_
c1 = slr_model.coef_[0]
#
# Learning exercises:
#  1. What do you expect if we use a larger value of the "fluctuations"?
#   what if we use a smaller one?
#   what if we use fluctuations = 0?
#  2. How to plot a line + points?
#
#
# Multiple Linear Regression - MLR
#
# Dataset
# Features
X1_mlr = X  # in [1, 3]
X2_mlr = 2 * np.random.random(size=(obs_number, 1))  # in [0, 2]
X3_mlr = 4 + 3 * np.random.random(size=(obs_number, 1))  # in [4, 7]
# bring them together
X_mlr = np.concatenate((X1_mlr, X2_mlr, X3_mlr), axis=1)
# Plot the features alone
if plotting:
    plt.plot(X1_mlr.squeeze(), '+')
    plt.suptitle('X1', fontsize=16)
    plt.show()
    plt.plot(X2_mlr.squeeze(), '+')
    plt.suptitle('X2', fontsize=16)
    plt.show()
    plt.plot(X3_mlr.squeeze(), '+')
    plt.suptitle('X3', fontsize=16)
    plt.show()
    plt.plot(X1_mlr.squeeze(), 'r+')
    plt.plot(X2_mlr.squeeze(), 'b*')
    plt.plot(X3_mlr.squeeze(), 'go')
    plt.suptitle('X1, X2, X3', fontsize=16)
    plt.show()
# Target
c0_mlr, c1_mlr, c2_mlr, c3_mlr = 5, 8, -5, 0.1
y_mlr = c0_mlr + c1_mlr * X1_mlr.squeeze() + c2_mlr * X2_mlr.squeeze() + c3_mlr * X3_mlr.squeeze() + \
        fluctuations * np.random.randn(obs_number)
if plotting:
    plt.plot(X1_mlr.squeeze(), y_mlr, 'b+')
    plt.suptitle('Y vs X1', fontsize=16)
    plt.show()
    plt.plot(X2_mlr.squeeze(), y_mlr, 'b+')
    plt.suptitle('Y vs X2', fontsize=16)
    plt.show()
    plt.plot(X3_mlr.squeeze(), y_mlr, 'b+')
    plt.suptitle('Y vs X3', fontsize=16)
    plt.show()
# Model
mlr_model = LinearRegression()
# train
mlr_model.fit(X_mlr, y_mlr)
c0_mlr_hat = mlr_model.intercept_
c1_mlr_hat = mlr_model.coef_[0]
c2_mlr_hat = mlr_model.coef_[1]
c3_mlr_hat = mlr_model.coef_[2]
# Results
print('###  MLR  ###')
print('Let\'s have a look at the MLR model results')

print(f'source c0 = {c0_mlr:.2f} \t model c0 = {c0_mlr_hat:.2f} --> relative difference {100*(c0_mlr-c0_mlr_hat)/c0_mlr_hat:.2f}%')
print(f'source c1 = {c1_mlr:.2f} \t model c0 = {c1_mlr_hat:.2f} --> relative difference {100*(c1_mlr-c1_mlr_hat)/c1_mlr_hat:.2f}%')
print(f'source c2 = {c2_mlr:.2f} \t model c0 = {c2_mlr_hat:.2f} --> relative difference {100*(c2_mlr-c2_mlr_hat)/c2_mlr_hat:.2f}%')
print(f'source c3 = {c3_mlr:.2f} \t model c0 = {c3_mlr_hat:.2f} --> relative difference {100*(c3_mlr-c3_mlr_hat)/c3_mlr_hat:.2f}%')

# Predictions
# Learning exercises:
# 1. Understand why and how this works:
i = 0
feature_numb = np.shape(X_mlr)[1]
X_pred_mlr = np.empty([new_obs, feature_numb])
for feature in [X1_mlr, X2_mlr, X3_mlr]:
    first, last = feature.min(), feature.max()
    values = np.linspace(first, last, new_obs)
    X_pred_mlr[:, i] = values
    i += 1
y_mlr_pred = mlr_model.predict(X=X_pred_mlr)
# 2. repeat model generation with different values of source data (both coefficients and fluctuations)
# 3. improve the plots (x, y, title, colours, markers, subplots)
# 4. try with some real data, e.g. height and weight, weight and gender, major and mark, ...
#
# Model Evaluation
#
# k-fold CV doc:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
K = 5
error = 'neg_mean_squared_error'  # recall: MSE = RSS/N
# choose score from https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
results_slr = cross_validate(slr_model, X, y, cv=K, scoring=error)
results_mlr = cross_validate(mlr_model, X_mlr, y_mlr, cv=K, scoring=error)

print('So long.')
