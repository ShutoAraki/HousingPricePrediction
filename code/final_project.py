#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Apr 16 16:04:30 2018
    Last Edited on Thu May 10 16:12 2018
    
    DePauw University Spring 2018
    CSC396: Data Mining FINAL PROJECT
    
    Special thanks to Professor Steve Bogaerts for offering this course for
    the first time and introducing us to the field of data science and machine 
    learning. We had a wonderful time.
    
    @author: Shuto Araki and Taras Tataryn
    
    @dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    
    Various feature preprocessing techniques and different types of regression
    algorithms are explored.
    
    ======= GLOBAL VARIABLES =======
    
    originalDF:
        The training dataset that is read from train.csv in data folder
    
    X, y: 
        The training attributes and SalePrice, all preprocessed
    
    RMSE: 
        Scorer for cross_val_score function
    
    
    ======= FUNCTIONS =======
    
    knowTarget(originalDF):
        Displays summary data and distribution of SalePrice.
        
    checkCorr(originalDF):
        Displays heat map of correlations among attributes.
        
    featureEngineering(originalDF):
        Returns a DataFrame with engineered features added.
        
    checkOutliers(originalDF):
        Displays scatter plots to see which houses might be considered outliers
    
    dropOutliers(originalDF):
        Returns a DataFrame with specified houses (that we thought are 
        outliers) dropped from the originalDF
        
    checkMissingValues(originalDF):
        Displays a summary of what attributes are missing by how much.
    
    handleMissingByImputer(originalDF):
        Returns a training attributes (X) with their missing values filled
        with the mean.
        
    standardize(originalDF):
        Returns a DataFrame with all the values standardized.
    
    testResultsData(originalDF):
        Tests which preprocessing strategy performs well.
        
    preprocess(originalDF):
        Returns a DataFrame with all the data preprocessed (handling missing values, 
        adding engineered features, taking care of correlated attributes, etc.)
        
    mean_squared_error_(originalDF):
        Calculates MSE. This function is to make the scoring in cross_val_score 
        function look cleaner because there is no parameter for RMSE calculation
        in this built-in function.
    
    tuneGBR(originalDF):
        Rough hyperparamter tuning of Gradient Boosting Regressor. Finer tuning
        and other algorithms are tuned using XSEDE allocation with 120GB of memory.
        Refer to xesede1.py, xsede2.py, ..., xsede5.py, and xsedeWeights.py for
        more tunings.
    
    tryModel(modelName):
        Runs a model and displays its RMSE with 10-fold cross validation and 
        random_state = 0.
        
    tryStackingModels(modelNames, metaModelName):
        Takes an array of model names in string and a meta model name in string.
        Displays its RMSE with 10-fold cross validation.
        
    tryAveragingModels(modelNames, pWeights):
        Takes an array of model names in string and a list of weights on each model.
        Displays its RMSE with 10-fold cross validation.
    
    submitResult(originalDF):
        Creates a csv file of prediction from testing dataset provided by Kaggle
        to submit to the competition.
        
"""

# Import Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from mlxtend.regressor import StackingRegressor


# Read Data
path_to_train = 'data/train.csv'
originalDF = pd.read_csv(path_to_train)


# Understanding the target variable SalePrice
def knowTarget(originalDF):
    target = originalDF.loc[:, "SalePrice"]
    
    print(target.describe())
    # BEGIN: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Seaborn library supports historgram and show the
    # distributions of data.
    sns.distplot(target)
# END: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


def checkCorr(originalDF):
    """
    Checking Highly Correlated Attributes
    """
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: This code visualizes the correlation matrix of the data
    # using heatmap, representing different correlation coefficients by
    # different colors.
    corrmat = originalDF.corr()
    f, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    
    #Zoom in the important variables
    #saleprice correlation matrix
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(originalDF[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    
    """
    It seems like 1stFlrSF and TotalBsmtSF,
    TotRmsAbvGr and GrLivArea, YearBuilt and GarageYrBlt,
    GarageArea and GarageCars are highly correlated respectively.
    Let us check the specific correlations.
    """
    cor1 = originalDF.loc[:, "1stFlrSF"].corr(originalDF.loc[:, "TotalBsmtSF"])
    cor2 = originalDF.loc[:, "TotRmsAbvGrd"].corr(originalDF.loc[:, "GrLivArea"])
    cor3 = originalDF.loc[:, "YearBuilt"].corr(originalDF.loc[:, "GarageYrBlt"])
    cor4 = originalDF.loc[:, "GarageArea"].corr(originalDF.loc[:, "GarageCars"])
    
    print("1st Floor SF and Total Basement SF")
    print(cor1)
    print("Total Rooms Above Ground and Ground Living Area")
    print(cor2)
    print("Year Built and Garage Year Built")
    print(cor3)
    print("Garage Area and Garage Cars")
    print(cor4)

# Maybe try dropping those with abs(corr) > 0.9?


def featureEngineering(originalDF):
    #replace yearBuilt with Years Old (2008 - the year)
    yearBuilt = originalDF.loc[:, "YearBuilt"]
    # 2008 because the data was collect in 2006-2010
    originalDF.loc[:, "YearsOld"] = 2008 - yearBuilt
    
    #create new column called Yard Size (Lot Area - Ground Floor Size)
    lotArea = originalDF.loc[:, "LotArea"]
    gFlrSize = originalDF.loc[:, "GrLivArea"]
    originalDF.loc[:, "YardSize"] = lotArea - gFlrSize
    
    # BEGIN: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    # EXPLANATION: This code utilizes the LabelEncoder from the sklearn.preprocessing 
    # package. For our purposes its used to transform non-numerical values (i.e. 
    # KitchenQual which has Typical/Average, Excellent, etc. as values and may not be 
    # interpreted well in our predictions)
    lblEncodeList = ["KitchenQual", "GarageQual", "BsmtQual", "BsmtCond", "ExterQual", "GarageCond",
                     "OverallCond", "FireplaceQu", "ExterCond", "HeatingQC", "BsmtFinType1", "BsmtFinType2"]
	# Label Encoder
    for column in lblEncodeList:
        lbl = LabelEncoder()
        lbl.fit(list(originalDF[column].values))
        originalDF[column] = lbl.transform(list(originalDF[column].values))
    # END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    
    return originalDF


def checkOutliers(originalDF):
    originalDF = featureEngineering(originalDF)
    
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Concatinated dataframes are plotted using
    #bivariate analysis saleprice/grlivarea
    vsGrLivArea = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'GrLivArea']], axis=1)
    vsGrLivArea.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    
    vsLotArea = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'LotArea']], axis=1)
    vsLotArea.plot.scatter(x='LotArea', y='SalePrice', ylim=(0,800000))
    
    """
    It seems like two points are way out of the trend in the graph.
    The two points in the upper part follow the trend, but they still
    look like they have something abnormal.
    """
    vsYearsOld = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'YearsOld']], axis=1)
    vsYearsOld.plot.scatter(x='YearsOld', y='SalePrice', ylim=(0,800000))
    """
    A point in the middle right seems too expensive for an old house. Maybe?
    The performance will tell us whether to drop it or not.
    """
    
    vsLotFrontage = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'LotFrontage']], axis=1)
    vsLotFrontage.plot.scatter(x='LotFrontage', y='SalePrice', ylim=(0,800000))
    #lotFrontageOutlier = (originalDF.loc[:, "LotFrontage"] > 300) & (originalDF.loc[:, "SalePrice"] > 100000)
    vsTotalBsmtSF = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'TotalBsmtSF']], axis=1)
    vsTotalBsmtSF.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
#    totalBsmtSFOutlier = (originalDF.loc[:, "TotalBsmtSF"] > 6000)
    # this outlier was same as one discovered above
    vsKitchenAbvGr = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'KitchenAbvGr']], axis=1)
    vsKitchenAbvGr.plot.scatter(x='KitchenAbvGr', y='SalePrice', ylim=(0,800000))
    KitchenAbvGrOutlier = (originalDF.loc[:, "KitchenAbvGr"] < 0.5)
    print(originalDF.loc[KitchenAbvGrOutlier])
    #id number 955 value = 0, definitely an outlier
    vsBedroomAbvGr = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'BedroomAbvGr']], axis=1)
    vsBedroomAbvGr.plot.scatter(x='BedroomAbvGr', y='SalePrice', ylim=(0,800000))
#    BedroomAbvGrOutlier = (originalDF.loc[:, "BedroomAbvGr"] > 7)
    
    # Identifying the IDs of the two (or four) outliers from GrLivArea
    print(originalDF.sort_values(by = 'GrLivArea', ascending = False)[:4])
    # Identifying the IDs of the four outliers from vsLotArea
    print(originalDF.sort_values(by = 'LotArea', ascending = False)[:4])
    # Identifying the IDs of the one outlier from YearsOld
    yearsOldOutlier = (originalDF.loc[:, "YearsOld"] > 120) & (originalDF.loc[:, "SalePrice"] > 400000)
    print(originalDF.loc[yearsOldOutlier])


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
# potentially delete PoolQC/MiscFeature/Alley (greater than 90% missing values)


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
# Fine tuning is done using the XSEDE allocation. Refer to xsede1.py, esede2.py, etc.
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
          # BEGIN: https://www.kaggle.com/gmishrakec/multi-regression-techniques/code
          # EXPLANATION: Since Lasso regression is very sensitive to outliers, 
          # it is better to use RobustScaler object to make it robust.
          "lasso": make_pipeline(RobustScaler(), Lasso(alpha =0.0005, 
                                                       random_state=0)),
          # END: https://www.kaggle.com/gmishrakec/multi-regression-techniques/code
          "bayridge": BayesianRidge(), 
          "ridge": Ridge()
         }

# This function takes a model name in string
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


# This function takes an array of model names in string, and a meta model name in string
def tryStackingModels(modelNames, metaModelName):
    X, y = preprocess(originalDF)
    regs = [models[modelName] for modelName in modelNames]
    meta = models[metaModelName]
    
    stack = StackingRegressor(regressors = regs, 
                              meta_regressor = meta)

    cvScores = model_selection.cross_val_score(stack, X, y, cv=10, scoring = RMSE)
    
    print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))


# This function takes a model name in string
def tryAveragingModels(modelNames, pWeights):
    try:
        X, y = preprocess(originalDF)
        algs = tuple(models[modelName] for modelName in modelNames)
        # This gives an equal weight on every model
#        averaged_models = AveragingModels(models = algs, weights = [1/len(algs) for i in range(len(algs))])
        averaged_models = AveragingModels(models = algs, weights = pWeights)
        cvScores = model_selection.cross_val_score(averaged_models, X, y, cv=10, scoring=RMSE)
        
        print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))
    except:
        print("The specified models do not exist in the models.")


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
    # Standardize the Trainingset
    trainDF = standardize(trainDF)
    
    """ # Weighted Average
    modelNames = ["gbr", "xgb", "lgb", "lasso", "ridge"]
    algs = tuple(models[modelName] for modelName in modelNames)
    model = AveragingModels(models = algs, weights = [0.24, 0.07, 0.14, 0.46, 0.09])
    """
    """
    modelNames = ["lgb", "lasso"]
    metaModelName = "ridge"
    regs = [models[modelName] for modelName in modelNames]
    meta = models[metaModelName]
    model = StackingRegressor(regressors = regs, 
                              meta_regressor = meta)
    """
    model = models["ridge"]

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

