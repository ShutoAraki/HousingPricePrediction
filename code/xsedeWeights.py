#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Apr 16 16:04:30 2018
    
    Data Mining Final Project
    
    @author: Shuto Araki and Taras Tataryn
    
    @dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    
    Various feature engineering techniques and different types of regression
    algorithms will be explored in the final version.
    
"""

# Import Dependencies
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, LabelEncoder, RobustScaler
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, BayesianRidge, Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Read Data
path_to_train = 'data/train.csv'
originalDF = pd.read_csv(path_to_train)


def featureEngineering(originalDF):
    #replace yearBuilt with Years Old (2008 - the year)
    yearBuilt = originalDF.loc[:, "YearBuilt"]
    # 2008 because the data was collect in 2006-2010
    originalDF.loc[:, "YearsOld"] = 2008 - originalDF.loc[:, "YearBuilt"]
    
    #create new column called Yard Size (Lot Area - Ground Floor Size)
    lotArea = originalDF.loc[:, "LotArea"]
    gFlrSize = originalDF.loc[:, "GrLivArea"]
    originalDF.loc[:, "YardSize"] = lotArea - gFlrSize
    totalSF = (originalDF["GrLivArea"] + originalDF["TotalBsmtSF"])
    lblEncodeList = ["KitchenQual", "GarageQual", "BsmtQual", "BsmtCond", "ExterQual", "GarageCond",
                     "OverallCond", "FireplaceQu", "ExterCond", "HeatingQC", "BsmtFinType1", "BsmtFinType2"]
	# Label Encoder
    for column in lblEncodeList:
        lbl = LabelEncoder()
        lbl.fit(list(originalDF[column].values))
        originalDF[column] = lbl.transform(list(originalDF[column].values))
    
    return originalDF


def dropOutliers(originalDF):
    # Deleting the outliers
    rowsToDrop = (originalDF.loc[:, "Id"] == 1299) | (originalDF.loc[:, "Id"] == 524) | (originalDF.loc[:, "Id"] == 1183) | (originalDF.loc[:, "Id"] == 692) | (originalDF.loc[:, "Id"] == 186) | (originalDF.loc[:, "Id"] == 314) | (originalDF.loc[:, "Id"] == 336) | (originalDF.loc[:, "Id"] == 250) | (originalDF.loc[:, "Id"] == 707) 
    
    originalDF = originalDF.drop(originalDF[rowsToDrop].index)
    
    return originalDF


def checkMissingValues(originalDF):
    total = originalDF.isnull().sum().sort_values(ascending=False)
    percent = (originalDF.isnull().sum()/originalDF.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))
## potentially delete PoolQC/MiscFeature/Alley (greater than 90% missing values)


def handleMissingByImputer(X):
    # Dealing with "All missing case"
    for col in X.columns:
        if X.loc[:, col].isnull().all():
            X.loc[:, col] = 0
    
    # BEGIN: from https://www.kaggle.com/dansbecker/handling-missing-values
    # EXPLANATION: Imputers can fill in missing values with the mean by default.
    # Filling in missing values is usually better than dropping them entirely
    # because those that are not missing might indicate some valuable patterns.
    my_imputer = Imputer()
    X = my_imputer.fit_transform(X)
    # END: from https://www.kaggle.com/dansbecker/handling-missing-values
    return X


def standardize(df):
    
    someCols = df.select_dtypes(exclude=['object']).columns
    
    stds = df.loc[:, someCols].std()
    mean = df.loc[:, someCols].mean()
    
    df.loc[:, someCols] = (df.loc[:, someCols] - mean) / stds
    
    return df


# Try and experiment with data with different preprocessing
def testResultsData(originalDF):
    
    # Build and test Grandient Boosting Regressor
    def testGBR(df, n_estimators=100, max_depth=3):
        y = df.loc[:, "SalePrice"]
        X = df.drop(["SalePrice"], axis=1)
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
        
        X = handleMissingByImputer(X)
        
        cvScores = model_selection.cross_val_score(gbr, X, y, cv=10,
                                                   scoring = 'neg_mean_squared_error')
        print("\tMean Squared Error:", -1 * cvScores.mean())
        
    
    # Build and test Decision Tree Regressor (a lot more tweaking to do)
    def testRF(df):
        y = df.loc[:, "SalePrice"]
        X = df.drop(["SalePrice"], axis=1)
        rf = RandomForestRegressor()
        
        X = handleMissingByImputer(X)
        
        cvScores = model_selection.cross_val_score(rf, X, y, cv=10,
                                                   scoring = 'neg_mean_squared_error')
        print("\tMean Squared Error:", -1 * cvScores.mean())
    
    # Add Engineered Features
    originalDF = featureEngineering(originalDF)
    # Dropping Outliers
    originalDF = dropOutliers(originalDF)
    # Standardize the Data
    originalDF.loc[:, originalDF.columns != "SalePrice"] = standardize(originalDF.drop(["SalePrice"], axis=1))
    """
    Preparing Different Encoding Strategies and Attribute Selection
    """
    ## One Hot Encoding
    colNames = ['SalePrice', 'LotArea', 'Neighborhood','OverallQual','GrLivArea','FullBath','GarageCars','YardSize','YearsOld']
    trainPredictors = originalDF.loc[:, colNames]
    selected_hot_end_DF = pd.get_dummies(trainPredictors)
    # Dropping attributes that have 90% missing values
    most_hot_end_DF = pd.get_dummies(originalDF.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1))
    # Dropping attributes that have 90% missing values or are highly correlated with other variables
    corr_dropped_DF = pd.get_dummies(originalDF.drop(['PoolQC', 'MiscFeature', 'Alley', 'GarageArea', 'YearBuilt'], axis=1))
    
    print("Graident Boosting Regressor: n_estimators = 100")
    print("\tSelected Columns")
    testGBR(selected_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values")
    testGBR(most_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testGBR(corr_dropped_DF)
    
    print()
    # The MSE is reduced by changing the n_estimators parameter
    print("Graident Boosting Regressor: n_estimators = 200")
    print("\tAll Attributes except for ones that have many missing values")
    testGBR(most_hot_end_DF, n_estimators=200)
    print("\tAll Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testGBR(corr_dropped_DF, n_estimators=200)
    
    print()
    
    print("Random Forest Regressor")
    print("\tSelected Columns")
    testRF(selected_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values")
    testRF(most_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testRF(corr_dropped_DF)
    

# One-Hot Encoding seems like the best data preprocessing
def preprocess(originalDF):
    # Add Engineered Features
    originalDF = featureEngineering(originalDF)
    # Dropping Outliers
    originalDF = dropOutliers(originalDF)
    # Standardize the Data
    originalDF.loc[:, originalDF.columns != "SalePrice"] = standardize(originalDF.drop(["SalePrice"], axis=1))
    
    # Dropping attributes that have 90% missing values
    most_hot_end_DF = pd.get_dummies(originalDF.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1))
    
    y = most_hot_end_DF.loc[:, "SalePrice"]
    X = most_hot_end_DF.drop(["SalePrice"], axis=1)
    X = handleMissingByImputer(X)
    y = np.log1p(y)
    
#    # Principal Component Analysis
#    pca = PCA(whiten=True)
#    pca.fit(X)
#    X = pca.transform(X)
    
    return X, y

# Store the clean X and y as global variables
X, y = preprocess(originalDF)


# BEGIN: https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/single_model/base_model.py
# EXPLANATION: This method creates a scorer for the GridSearchCV so that
# the search evaluates the model's performance using RMSE
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)
# END: https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/single_model/base_model.py


# Rough hyperparameter tuning (one parameter at a time).
# Fine tuning is done using the XSEDE allocation. Refer to xsede1.py and xsede2.py
def tuneGBR(df):
    X, y = preprocess(df)
    """
    param_test1 = {'min_samples_split':range(2,20)}
    gsearch1 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 1,
                max_depth = 4,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test1, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch1.fit(X, y)
    print("Best param and its score", gsearch1.best_params_, -gsearch1.best_score_, sep='\n')
    """
    """
    param_test2 = {'min_samples_leaf':range(2,10)}
    gsearch2 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_split = 17,
                max_depth = 4,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test2, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch2.fit(X, y)
    print("Best param and its score", gsearch2.best_params_, -gsearch2.best_score_, sep='\n')
    """
    """
    param_test3 = {'max_depth':range(2,10)}
    gsearch3 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 3,
                min_samples_split = 17,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test3, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch3.fit(X, y)
    print("Best param and its score", gsearch3.best_params_, -gsearch3.best_score_, sep='\n')
    """
    
    param_test4 = {'subsample': [0.76, 0.78, 0.8, 0.82, 0.84]}
    gsearch4 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 3,
                min_samples_split = 17,
                max_depth = 5, 
                max_features = 'sqrt',
                random_state = 0
                ), 
            param_grid = param_test4, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch4.fit(X, y)
    print("Best param and its score", gsearch4.best_params_, -gsearch4.best_score_, sep='\n')
    

# Dictionary of Different Models
models = {
          # GBR is tuned using the result obtained from xsede1.py
          "gbr": GradientBoostingRegressor(n_estimators=1000,
                                           learning_rate=0.01, 
                                           min_samples_split=17, 
                                           min_samples_leaf=7, 
                                           max_depth=5, 
                                           max_features='sqrt', 
                                           subsample=0.8, 
                                           random_state=0), 
          # XGB is tuned using the result obtained from xsede2.py
          "xgb": XGBRegressor(n_estimators=1000,
                              learning_rate=0.01, 
                              min_child_weight=3,
                              max_depth=6, 
                              gamma=0, 
                              subsample=0.8, 
                              colsample_bytree=0.85, 
                              random_state=0), 
          "lgb": LGBMRegressor(objective='regression', 
                               num_leaves=5, 
                               learning_rate=0.05, 
                               n_estimators=720, 
                               max_bin=55, 
                               bagging_fraction=0.8, 
                               bagging_freq=5, 
                               feature_fraction=0.2319, 
                               feature_fraction_seed=9, 
                               bagging_seed=9, 
                               min_data_in_leaf=6, 
                               min_sum_hessian_in_leaf=11,
                               random_state=0),
          "rf": RandomForestRegressor(n_estimators=1000, 
                                      bootstrap=True, 
                                      max_features='sqrt', 
                                      max_depth=6, 
                                      min_samples_split=3, 
                                      min_samples_leaf=1, 
                                      random_state=0), 
          "knn": KNeighborsRegressor(n_neighbors = 10), 
          "ada": AdaBoostRegressor(n_estimators=1000,
                                   learning_rate=0.01, 
                                   loss='square', 
                                   random_state=0),
          # BEGIN: 
          # EXPLANATION: 
          "lasso": make_pipeline(RobustScaler(), Lasso(alpha =0.0005, 
                                                       random_state=0)),
          # END: 
          "bayridge": BayesianRidge(), 
          "ridge": Ridge()
         }


def tryModel(modelName):
    try:
        alg = models[modelName]
        X, y = preprocess(originalDF)
        cvScores = model_selection.cross_val_score(alg, X, y, cv=10, scoring = RMSE)
    
        print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))
    except:
        print("The specified model does not exist in the models.")


# Inspired by: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# EXPLANATION: This class takes several different models as a tuple and the 
# final score is calculated by taking the weighted average of the predictions from 
# each model. The class was modified by Shuto Araki to calculate the WEIGHTED
# average instead of just the arithmetic average.
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        if len(weights) != len(models):
            print("ERROR: The number of models and weights mismatching.")
            return
        if sum(weights) != 1.0:
            print("The sum of weights must be 1.")
            return
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        #return np.mean(predictions, axis=1)
        predictionsDF = pd.DataFrame(data=predictions)
        weightedPredictions = predictionsDF.apply(lambda pred: pred * self.weights, axis=1)
        return np.sum(weightedPredictions, axis=1)
    
# END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard


# GBR, XGB, LGB --> 0.110463586
def tryAveragingModels(modelNames):
#    try:
        X, y = preprocess(originalDF)
        algs = tuple(models[modelName] for modelName in modelNames)
        averaged_models = AveragingModels(models = algs, weights = [1/len(algs) for i in range(len(algs))])
        
        cvScores = model_selection.cross_val_score(averaged_models, X, y, cv=10, scoring=RMSE)
        
        print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))
#    except:
#        print("The specified models do not exist in the models.")


# Create a csv file for submission
def submitResult(originalDF):
    
    print("File Name?")
    filename = input()
    
    # Read the test data
    testDF = pd.read_csv("data/test.csv")
    ids = testDF.loc[:, "Id"]
    
    # Dropping Outliers first
    originalDF = dropOutliers(originalDF)
    
    y = originalDF.loc[:, "SalePrice"]
    y = y.values
    trainDF = originalDF.drop(["SalePrice"], axis=1)
    
    # Concatinate test and original (= train)
    train_numRow = trainDF.shape[0]
    dataset = pd.concat(objs=[trainDF, testDF], axis=0)
    
    # One-Hot Encoding and Feature Engineering to create the full dataset
    dataset = featureEngineering(dataset)
    dataset = pd.get_dummies(dataset)
    
    # Splitting them again so that the number of features in
    # the test and original matches
    trainDF = dataset.iloc[:train_numRow, :].copy()
    testDF = dataset.iloc[train_numRow:, :].copy()
    
    # Add Engineered Features
    #trainDF = featureEngineering(trainDF)
    # Standardize the Trainingset
    trainDF = standardize(trainDF)
    
    modelNames = ["gbr", "xgb", "lgb", "lasso", "ridge"]
    algs = tuple(models[modelName] for modelName in modelNames)
    model = AveragingModels(models = algs, weights = [1/len(algs) for i in range(len(algs))])
    #model = StackingAveragedModels(base_models=algs, meta_model=models["lasso"])
    

    trainDF = handleMissingByImputer(trainDF)
    model.fit(trainDF, y)
    
    # Test dataset preprocessing
    testDF = standardize(testDF)
    testDF = handleMissingByImputer(testDF)
    
    predictions = model.predict(testDF)
    
    # Create submission file in csv
    submitDF = pd.DataFrame(
                            {
                            "Id": ids,
                            "SalePrice": predictions
                            }
                        )
    submitDF.to_csv("data/" + filename + ".csv", index = False)


# Should be lower than 0.105760
def model_test_weighting(df, n_jobs):
    X, y = preprocess(df)
    
    modelNames = ["gbr", "xgb", "lgb", "lasso", "ridge"]
    algs = tuple(models[modelName] for modelName in modelNames)
    stack = AveragingModels(models = algs, weights = [1/len(algs) for i in range(len(algs))])
    
    # Rough Search on weights
    param_grid = {
        'weights': [
                    [0.1, 0.1, 0.1, 0.1, 0.6], 
                    [0.1, 0.1, 0.1, 0.6, 0.1],
                    [0.1, 0.1, 0.6, 0.1, 0.1],
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.6, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.2, 0.5],
                    [0.1, 0.1, 0.2, 0.1, 0.5],
                    [0.1, 0.2, 0.1, 0.1, 0.5],
                    [0.2, 0.1, 0.1, 0.2, 0.5],
                    [0.1, 0.1, 0.1, 0.5, 0.2],
                    [0.1, 0.1, 0.2, 0.5, 0.1],
                    [0.1, 0.2, 0.1, 0.5, 0.1],
                    [0.2, 0.1, 0.1, 0.5, 0.1],
                    [0.1, 0.1, 0.5, 0.1, 0.2],
                    [0.1, 0.1, 0.5, 0.2, 0.1],
                    [0.1, 0.2, 0.5, 0.1, 0.1],
                    [0.2, 0.1, 0.5, 0.1, 0.1],
                    [0.1, 0.5, 0.1, 0.1, 0.2],
                    [0.1, 0.5, 0.1, 0.2, 0.1],
                    [0.1, 0.5, 0.2, 0.1, 0.1],
                    [0.2, 0.5, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.3, 0.4],
                    [0.1, 0.1, 0.3, 0.1, 0.4],
                    [0.1, 0.3, 0.1, 0.1, 0.4],
                    [0.3, 0.1, 0.1, 0.1, 0.4],
                    [0.1, 0.1, 0.2, 0.2, 0.4],
                    [0.1, 0.2, 0.1, 0.2, 0.4],
                    [0.2, 0.1, 0.1, 0.2, 0.4],
                    [0.1, 0.2, 0.2, 0.1, 0.4],
                    [0.2, 0.1, 0.2, 0.1, 0.4],
                    [0.2, 0.2, 0.1, 0.1, 0.4],
                    [0.1, 0.1, 0.1, 0.4, 0.3],
                    [0.1, 0.1, 0.3, 0.4, 0.1],
                    [0.1, 0.3, 0.1, 0.4, 0.1],
                    [0.3, 0.1, 0.1, 0.4, 0.1],
                    [0.1, 0.1, 0.2, 0.4, 0.2],
                    [0.1, 0.2, 0.1, 0.4, 0.2],
                    [0.2, 0.1, 0.1, 0.4, 0.2],
                    [0.1, 0.2, 0.2, 0.4, 0.1],
                    [0.2, 0.1, 0.2, 0.4, 0.1],
                    [0.2, 0.2, 0.1, 0.4, 0.1],
                    [0.1, 0.1, 0.4, 0.1, 0.3],
                    [0.1, 0.1, 0.4, 0.3, 0.1],
                    [0.1, 0.3, 0.4, 0.1, 0.1],
                    [0.3, 0.1, 0.4, 0.1, 0.1],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.2, 0.4, 0.1, 0.2],
                    [0.2, 0.1, 0.4, 0.1, 0.2],
                    [0.1, 0.2, 0.4, 0.2, 0.1],
                    [0.2, 0.1, 0.4, 0.2, 0.1],
                    [0.2, 0.2, 0.4, 0.1, 0.1],
                    [0.1, 0.4, 0.1, 0.1, 0.3],
                    [0.1, 0.4, 0.1, 0.3, 0.1],
                    [0.1, 0.4, 0.3, 0.1, 0.1],
                    [0.3, 0.4, 0.1, 0.1, 0.1],
                    [0.1, 0.4, 0.1, 0.2, 0.2],
                    [0.1, 0.4, 0.2, 0.1, 0.2],
                    [0.2, 0.4, 0.1, 0.1, 0.2],
                    [0.1, 0.4, 0.2, 0.2, 0.1],
                    [0.2, 0.4, 0.1, 0.2, 0.1],
                    [0.2, 0.4, 0.2, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.3, 0.3],
                    [0.1, 0.2, 0.1, 0.3, 0.3],
                    [0.2, 0.1, 0.1, 0.3, 0.3],
                    [0.1, 0.1, 0.3, 0.2, 0.3],
                    [0.1, 0.2, 0.3, 0.1, 0.3],
                    [0.2, 0.1, 0.3, 0.1, 0.3],
                    [0.1, 0.3, 0.1, 0.2, 0.3],
                    [0.1, 0.3, 0.2, 0.1, 0.3],
                    [0.2, 0.3, 0.1, 0.1, 0.3],
                    [0.3, 0.1, 0.1, 0.2, 0.3],
                    [0.3, 0.1, 0.2, 0.1, 0.3],
                    [0.3, 0.2, 0.1, 0.1, 0.3],
                    [0.1, 0.1, 0.3, 0.3, 0.2],
                    [0.1, 0.2, 0.3, 0.3, 0.1],
                    [0.2, 0.1, 0.3, 0.3, 0.1],
                    [0.1, 0.3, 0.1, 0.3, 0.2],
                    [0.1, 0.3, 0.2, 0.3, 0.1],
                    [0.2, 0.3, 0.1, 0.3, 0.1],
                    [0.3, 0.1, 0.1, 0.3, 0.2],
                    [0.3, 0.1, 0.2, 0.3, 0.1],
                    [0.3, 0.2, 0.1, 0.3, 0.1],
                    [0.1, 0.3, 0.3, 0.1, 0.2],
                    [0.1, 0.3, 0.3, 0.2, 0.1],
                    [0.2, 0.3, 0.3, 0.1, 0.1],
                    [0.3, 0.3, 0.1, 0.1, 0.2],
                    [0.3, 0.3, 0.1, 0.2, 0.1],
                    [0.3, 0.3, 0.2, 0.1, 0.1]
                   ]
    }
    model = model_selection.GridSearchCV(estimator=stack, param_grid=param_grid, n_jobs=n_jobs, cv=10, scoring=RMSE)
    model.fit(X, y)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
# Using the xxlarge jetstream image (44 vCPUs)
#model_test_weighting(originalDF, n_jobs=44)

# The result:
#    Best Params:
#    {'weights': [0.2, 0.1, 0.1, 0.5, 0.1]}
#    Best CV Score:
#    0.1054212705095865

def tune_precise_weighting(df, n_jobs):
    X, y = preprocess(df)
    
    modelNames = ["gbr", "xgb", "lgb", "lasso", "ridge"]
    algs = tuple(models[modelName] for modelName in modelNames)
    stack = AveragingModels(models = algs, weights = [0.2, 0.1, 0.1, 0.5, 0.1])
    
    weightList = []
    for first in range(-5, 5):
        for second in range(-5, 5):
            for third in range(-5, 5):
                for fourth in range(-5, 5):
                    for fifth in range(-5, 5):
                        if 0.2 + first/100 + 0.1 + second/100 + 0.1 + third/100 + 0.5 + fourth/100 + 0.1 + fifth/100 == 1:
                            weightList.append([0.2 + first/100, 
                                               0.1 + second/100, 
                                               0.1 + third/100, 
                                               0.5 + fourth/100, 
                                               0.1 + fifth/100])

    # Rough Search on weights
    param_grid = {
        'weights': weightList
    }
    model = model_selection.GridSearchCV(estimator=stack, param_grid=param_grid, n_jobs=n_jobs, cv=10, scoring=RMSE)
    model.fit(X, y)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
# Best Params:
#    {'weights': [0.24000000000000002, 0.07, 0.14, 0.46, 0.09000000000000001]}
#    Best CV Score:
#    0.10536595882184294

tune_precise_weighting(originalDF, 44)