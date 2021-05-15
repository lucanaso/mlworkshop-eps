# Author: Luca Naso, https://lucanaso.it/
# Creation date: 04 Nov 2020
# Contributors: empty for now - add your name once you start contributing
# License: GNU CPL v3.0 (see the LICENSE file on GitHub).
# Topic: supporting code for an introductory workshop on Machine Learning with Python

# ################################
# Programma di tutto il Workshop
# ################################
# 1. Introduzione al ML (definizione, esempi, tassonomia)
# 2. Dataset creation in Python
# 3. Problemi di Regressione (algoritmo: regressione lineare; formule e significati)
# 4. Valutazione dei modelli (panoramica, )
# 5. Problemi di Classificazione: solo cenni
# 6. Problemi non-supervisionati: solo cenni

# //====================>>>>>>>>>>
# Code begins
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error

# introduce a flag for showing / not showing the plots
plot = False
# ########################### #
# TOPIC 1: Introduzione al ML #
# ########################### #
#   --> only slides
#   end of topic 1
# ===========================/
# ######################### #
# TOPIC 2: Dataset creation #
# ######################### #
#   --> only code
#       2.1 Data
# ***************
# 2.1 Data
# ********
# Generate the data, both X and Y
# Set the seed to make results reproducible
np.random.seed(0)
#
# X
# Generate data about X first, X = independent variable = feature
# generate an array of points at random
obs_number = 200
X = 2 * np.random.random(size=(obs_number, 1))
# doc: https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
# - np.random.random = Returns random floats in the half-open interval [0.0, 1.0)
# can also use: X = 1 + 2 * np.random.random(obs_number)
#   this will create a 1D array, while we need a 2D array for building the model
# size=() sets the shape of the output array
# X is a column of size obs_number x 1, therefore it is a 2D array:
# np.ndim(X) = 2
#
# Y
# generate the values for the observations or target (Y)
# insert some linearity in the dataset
c0 = 2
c1 = 3
fluctuations = 1
y = c0 + c1 * X.squeeze() + fluctuations * np.random.randn(obs_number)
# - np.squeeze: Remove axes of length one from input, i.e. it makes X a row (from 2D to 1D).
#               It is needed to make the sum as a vector
#   np.ndim(X) = 2
#   np.ndim(X.squeeze) = 1
# doc: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
# - np.random.randn = Return a sample from the “standard normal” distribution.
# randn(10) returns 10 values from a "standard normal" distribution with sigma = 1 and mu = 0
# doc: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
#
# Plot the data
# Scatter plot of the observations
if plot:
    plt.plot(X.squeeze(), y, 'b+')
    plt.show()
# - plt.show() shows all of the figures/charts that have been created on a new window
# for colours and markers doc:
# doc: https://matplotlib.org/2.0.2/api/pyplot_api.html#matplotlib.pyplot.plot
#   end of topic 2
# ===========================/
#
# ################################ #
# TOPIC 3: Problemi di Regressione #
# ################################ #
#   --> slides + code
#       3.1 SLR: model, prediction, plot
#       3.2 MLR: model, prediction, plot
# *********************************************
# 3.1 SLR: model, prediction, plot
# ********************************
# Create the Model, for Linear Regression
lr_model = LinearRegression()
# i.e. create an instance of the object LinearRegression
# - we do not specify any property (attributes) for the object
# - this is just an empty object, with all the attributes and the methods specified in its class
# doc: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#
# Train, fit the model
# To train the model we use the method "fit" of the object.
# "train" means to compute the coefficients of the linear regression (via least squares)
lr_model.fit(X, y)
# this is it, the model is now complete and can be used for making predictions
#
# Predictions
# create data
# Now that we have our model we want to use it, i.e. we want to make predictions on new data
# So, first of all let's generate new data points (X_pred)
# we do that by generating new_obs points, evenly spaced between "first" and "last" obs:
new_obs = 100
first, last = X.min(), X.max()
X_pred = np.linspace(first, last, new_obs)[:, np.newaxis]
# - np.linspace: Returns evenly spaced numbers over a specified interval.
# doc: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# - np.newaxis: to add 1 dimension to an array
#   we need this because it is required when we want to make predictions
# doc: https://numpy.org/devdocs/reference/constants.html#numpy.newaxis
# Basic examples:
# a = np.linspace(0, 1, 100) produces an array with 1 dimension
#   np.shape(a) = (100,)
#   np.ndim(a) = 1
# b = np.linspace(0, 1, 100)[:, np.newaxis] produces an array with 2 dimensions
#   np.shape(b) = (100,1)
#   np.ndim(b) = 2
# in this case they contain the same data, we adding a "working/dummy" dimension
#
# Ready to make predictions
# use the method "predict" of the LinearRegression object
y_pred = lr_model.predict(X=X_pred)
# the predict method requires the X to have shape = (n_samples, n_features),
#                                   in our case we have 1 feature and 100 obs
# np.linspace(0, 1, 100) is a 1D array with 100 elements
# np.linspace(0, 1, 100)[:, np.newaxis] is a 2D array with 100 rows and 1 column
#
# Plot
# plot the data and the predictions (both of them with a scatterplot)
if plot:
    plt.plot(X.squeeze(), y, 'b+')  # observations used to train the model
    plt.plot(X_pred, y_pred, 'r-')  # predictions
    plt.show()  # plt.show() will put 2 plots in the same chart
#
# Result
# compare model coefficients with the "true" value
print('\n###   SLR   ###')
# model coefficients for the features are stored in the property "coef_" of the LinearRegression object
# model coefficient for the intercept is stored in the property "intercept_" of the LinearRegression object
a1 = lr_model.coef_[0]
a0 = lr_model.intercept_
print('The coefficient for our feature is {:.2f} (no unit)'.format(a1))
# print(f'The coefficient for our feature is {a1:.2f} (no unit)') # new style, "f-string", available from Python 3.6
print('The coefficient for the intercept is {:.2f} (same unit as the target)'.format(a0))
print('The model equation can then be written as:\n\t y = {:.2f} + {:.2f} * x'.format(a0, a1))
print('Let\'s compare the result with the model used to generate the data, i.e. with the "source":')
print('\t- feature coefficient:\t\t source = {:.2f}\t model = {:.2f}\t relative difference = {:.1f}%'.
      format(c1, a1, 100*(a1-c1)/a1))
print('\t- intercept coefficient:\t source = {:.2f}\t model = {:.2f}\t relative difference = {:.1f}%'.
      format(c0, a0, 100*(a0-c0)/a0))

# TODO (homework):
#  What do you expect if we use a larger value of the "fluctuations"?
#   what if we use a smaller one?
#   what if we use fluctuations = 0 ?
#  How to plot a line + points?

# *********************************************
# 3.2 MLR: model, prediction, plot
# ********************************
# MLR means that we have more than 1 feature
#
# Data
# let's create a dataset where each observations has 3 features
X1_mlr = X  # X1 is in the range [0, 2)
X2_mlr = 4 + 3 * np.random.random(size=(obs_number, 1))  # X2 is in the range [4, 7)
X3_mlr = 1 + 4 * np.random.random(size=(obs_number, 1))  # X3 is in the range [1, 5)
# Plot each feature to check their ranges
if plot:
    plt.plot(X1_mlr.squeeze(), 'b+')
    plt.suptitle('X1', fontsize=16)
    plt.show()
    plt.plot(X2_mlr.squeeze(), 'b+')
    plt.suptitle('X2', fontsize=16)
    plt.show()
    plt.plot(X3_mlr.squeeze(), 'b+')
    plt.suptitle('X3', fontsize=16)
    plt.show()
# - plt.suptitle sets the plot title
#
# Now we change the target and make it depends on the 3 features
c0_mlr, c1_mlr, c2_mlr, c3_mlr = 2, 7, -4, 2
# c0, c1, c2, c3 = 1, 1, 1, 1
fluctuations = 1
y_mlr = c0_mlr + c1_mlr * X1_mlr.squeeze() + c2_mlr * X2_mlr.squeeze() + c3_mlr * X3_mlr.squeeze() + \
        fluctuations * np.random.randn(obs_number)
# To plot y vs all the features we need a 4D plot (no way)
# we plot it one by one, e.g. y vs X1, y vs X2, y vs X3
if plot:
    plt.plot(X1_mlr.squeeze(), y_mlr, 'b+')
    plt.suptitle('Y vs X1', fontsize=16)
    plt.show()
    plt.plot(X2_mlr.squeeze(), y_mlr, 'g+')
    plt.suptitle('Y vs X2', fontsize=16)
    plt.show()
    plt.plot(X3_mlr.squeeze(), y_mlr, 'r+')
    plt.suptitle('Y vs X3', fontsize=16)
    plt.show()
#
# Model
mlr_model = LinearRegression()
# Let's put X1, X2 and X3 together into a single variable (array).
#   we want each row of the new variable to be an observation
#   ==> 1 row = 3 values = 3 features = (X1, X2, X3)
features = np.concatenate((X1_mlr, X2_mlr, X3_mlr), axis=1)
# - np.concatenate: Join a sequence of arrays along an existing axis.
# axis is integer and starts from 0.
# doc: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
# - np.shape(features) = (200, 3) --> features is an NdArray of 200 items, each item is a list of 3 elements.
# row 1 = features[0, :] = features[0]
#
# Train, fit the model
# Fit works the same way as SLR, but now "X" (here called "features") has 3 columns
mlr_model.fit(features, y_mlr)
#
# Predictions
# let's create the data to make predictions
new_obs = 100
# for each observations we need the values for 3 features
# for each feature we need new_obs values
# we want 100 x 3 NdArray
# total number of features
feature_numb = np.shape(features)[1]  # 3
# Initialise the array for the predictions
X_preds_mlr = np.empty([new_obs, feature_numb])
# - np.empty: Return a new array of given shape and type, without initializing entries.
# doc: https://numpy.org/doc/stable/reference/generated/numpy.empty.html
i = 0
for feature in [X1_mlr, X2_mlr, X3_mlr]:
    first, last = feature.min(), feature.max()
    values = np.linspace(first, last, new_obs)
    X_preds_mlr[:, i] = values
    i += 1
# we make predictions as before, but now X has 3 columns (the target is still just 1)
y_mlr_pred = mlr_model.predict(X=X_preds_mlr)
# plot predictions for each feature
#   be very careful on how you interpret these plots.
#       in each plot only 1 feature is shown, BUT the other 2 features are NOT CONSTANT!
if plot:
    plt.plot(X1_mlr.squeeze(), y_mlr, 'b+')
    plt.plot(X_preds_mlr[:, 0].squeeze(), y_mlr_pred, 'k.')
    plt.suptitle('MLR model', fontsize=16)
    plt.xlabel('X1', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.show()
    plt.plot(X2_mlr.squeeze(), y_mlr, 'g+')
    plt.plot(X_preds_mlr[:, 1].squeeze(), y_mlr_pred, 'k.')
    plt.suptitle('MLR model', fontsize=16)
    plt.xlabel('X2', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.show()
    plt.plot(X3_mlr.squeeze(), y_mlr, 'r+')
    plt.plot(X_preds_mlr[:, 2].squeeze(), y_mlr_pred, 'k.')
    plt.suptitle('MLR model', fontsize=16)
    plt.xlabel('X3', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.show()

# Result
# compare model coefficients with the "true" value
print('\n###   MLR   ###')
print('Let\'s have a look at the MLR model results.')
for a, c in zip(mlr_model.coef_, [c1_mlr, c2_mlr, c3_mlr]):
    print('\t- feature coefficient:\t source = {:.2f}\t model = {:.2f}\t relative difference = {:.1f}%'.
          format(c, a, 100 * (a - c) / a))
a0_mlr = mlr_model.intercept_
print('\t- intercept coeffic.:\t source = {:.2f}\t model = {:.2f}\t relative difference = {:.1f}%'.
      format(c0_mlr, a0_mlr, 100 * (a0_mlr - c0_mlr) / a0_mlr))
#
# TODO (homework):
#   - non ordinare i valori delle feature nelle osservazioni, sia in X sia in X_preds inserirli invece a caso
#   - repeat models with different values of source data (both coefficients and fluctuations)
#   - improve the plots (x, y, title, colours, markers, subplots)
#   - try with some real data, e.g. height and weight, weight and gender, major and mark, ...
#
# end of topic 3
# ===========================/

# ################################ #
# TOPIC 4. Valutazione dei modelli #
# ################################ #
#   --> slides + code
#       4.1 Validation set
#       4.2 K-fold CV
#
# ************************************
# 4.1 Validation set
# ******************
# Evaluate the model
# we have to split the dataset, use some data to train a model, and the set-aside data to test it
#
# 1. Split the data
print('\n###   Model Evaluation   ###')
print('1. Evaluate with validation set (Single 75-25 random split):')
# why bother doing all the steps by ourselves?
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.75, random_state=1)
# - train_test_split (from sklearn.model_selection) creates the 2 datasets for us
#   it "splits arrays or matrices into random train and test subsets"
# doc: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#
# now that we have split the data, we need to use:
#   - train dataset to train the model
#   - test dataset to evaluate results
# 2. train/create the model
lr_model2 = LinearRegression()
lr_model2.fit(X_train, y_train)
# 3. Make predictions
# on the test dataset (not used for the train)
X_pred2 = X_test
y_pred2 = lr_model2.predict(X=X_pred2)
# Plot
# plot the data and the predictions
if plot:
    plt.plot(X_train.squeeze(), y_train, 'b+')  # observations used to train the model
    plt.plot(X_pred2, y_pred2, 'r-')  # predictions on test-data
    # TODO Homework:
    #   - compare with previous predictions and models
    # plt.plot(X_pred, y_pred, 'g-')  # Previous predictions and model
    plt.show()
# Coefficient values
# compare model coefficients with the "true" value
print('Validation for the SLR model')
a1_val = lr_model2.coef_[0]
a0_val = lr_model2.intercept_
print('\tThe coefficient for our feature is {:.2f} (no unit), before it was {:.2f}'.format(a1_val, a1))
print('\tThe coefficient for the intercept is {:.2f} (same unit as the target), before it was {:.2f}'.format(a0_val, a0))
print('\tThe model equation can then be written as:\n\t y = {:.2f} + {:.2f} * x'.format(a0_val, a1_val))
print('\tLet\'s compare the result with the model used to generate the data, i.e. with the "source":')
print('\t\t- feature coefficient:\t source = {:.2f}\t\t model = {:.2f}\t relative difference = {:.1f}%'.
      format(c1, a1_val, 100*(a1_val-c1)/a1_val))
print('\t\t- intercept coeffici.:\t source = {:.2f}\t model = {:.2f}\t relative difference = {:.1f}%'.
      format(c0, a0_val, 100*(a0_val-c0)/a0_val))
#
# 3. Compute error
# Rsquared = lr_model2.score(X_test, y_test)
# The .score method returns the coefficient of determination R^2 of the prediction
# print('\tR-squared \t\t\t= {:.2f}%'.format(100*Rsquared))
# We can compute the Mean Squared Error (MSE), with the function "mean_squared_error":
mse2 = mean_squared_error(y_test, y_pred2)
print('\n\tMean Squared Error \t= {:.3f}'.format(mse2))

# with random_state = 1 we get:
# R2 = 74.78%
# MSE = 0.91
# Random State --> MSE
# 0 --> 0.910
# 1 --> 0.958
# 2 --> 0.962
# 3 --> 0.89
# 4 --> 0.87
#
# TODO (Homework):
#   - repeat using different values of random_state
#       - create a plot with MSE vs random_state
#   - repeat using different values of train_size
#       - create a plot with MSE vs train_size
#   - use Validation set for MLR

# **************************
# 4.2 K-fold CV
# *************
print('\n2. Evaluate with K-fold CV:')
print('Validation for the SLR model')
# to perform k-fold Cross-Validation we use cross_validate (still from sklearn.model_selection)
# doc: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
# it returns a dictionary with several values
# it requires the following input:
# - model
# - Dataset: observations + targets (all of them, no split)
# - number of folds, i.e. K value
K = 5
# - score to be computed, i.e. the metric for measuring the error, e.g. MSE, R^2, ...
#   choose score from https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
error = 'neg_mean_squared_error'  # recall: MSE = RSS/N
# error = 'explained_variance'
# error = 'r2'
#   "Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R)."
#   ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
#
# Perform the k-fold cross validation
results = cross_validate(lr_model2, X, y, cv=K, scoring=error)
# to see the dictionary keys use:
# print(results.keys())
print('\tNumber of folds = {}'.format(K))
# error for each fold (need to change sign) and average error over the folds:
print('\tError for each fold: {}'.format(-results['test_score']))
print('\n\tError averaged on all folds: {:.3f}'.format(-results['test_score'].mean()))

# MSE values
#                       K=3         K=5         K=10
# random state = 0     0.917       0.931       0.936
# random state = 1     1.072       1.067       1.067
# random state = 2     1.142       1.116       1.093
# random state = 3     1.279       1.277       1.249
# random state = 4     1.052       1.062       1.053
# TODO (Homework):
#   repeat K-fold CV for different values of K, plot average-MSE vs K
#   repeat K-fold CV for different values of random state for each K, plot average-average-MSE vs K
#
# end of topic 4
# ===========================/
#
# #################################### #
# TOPIC 5. Problemi di Classificazione #
# #################################### #
#   --> slides + code
#       5.1 Example
#
# **********************
# 5.1 Example
# ***********
# Classification definition --> see slides
#
# Problem: Flower classification (iris)
# Dataset: iris
print('\n###   Classification example   ###')
from sklearn import datasets
#
print('Loading the iris dataset (observations + targets)')
iris = datasets.load_iris()
#
# import the class for the algorithm
from sklearn.svm import SVC
# create the object for the model
classifier = SVC()
#
print('Fitting the model with integers as target labels')
classifier.fit(iris.data, iris.target)
# iris.data (observations) has 150 observations and 4 features
# iris.target has 150 values for the targets
#
print('Predictions on the first 3 observations:')
print('\t', list(classifier.predict(iris.data[:3])))
#
print('Fitting the model using strings as target labels (before we used integers)')
classifier.fit(iris.data, iris.target_names[iris.target])
#
print('Predictions on the first 3 observations:')
print('\t', list(classifier.predict(iris.data[:3])))
# end of topic 5
# ===========================/
#
# #################################### #
# TOPIC 6. Problemi non-supervisionati #
# #################################### #
#   --> slides + code
#       6.1 Example
#
# **********************
# 6.1 Example
# ***********
# Clustering definition --> see slides
#
# Problem: data clustering
# Dataset: created
#
print('\n###   Clustering example   ###')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#
# create dataset
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.85, shuffle=True, random_state=0)
# n_samples = total number of observations
# centers = total number of groups/blobs
# cluster_std = "width" of each group
# doc: # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
#
# plot dataset
if plot:
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.show()
#
# create model instance
from sklearn.cluster import KMeans
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
# n_init = 10 --> repeat the whole procedure 10 times, each time with different initial conditions
# tol = tolerance --> to define when the method has converged
# run the model, i.e. return the estimated group for each observation
y_km = km.fit_predict(X)
#
# Plot results
# plot the 3 clusters
if plot:
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

print('See plots')
# end of topic 6
# ===========================/

# >>>>>>>>>>====================//
# Code ends
print('So long.')
