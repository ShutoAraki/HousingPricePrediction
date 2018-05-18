#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:04:30 2018

Data Mining Final Project Checkpoint

@author: Shuto Araki and Taras Tataryn

@dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Various feature engineering techniques and different types of regression 
algorithms will be explored in the final version.
For this first checkpoint, a few preprocessing techniques and Gradient Boosting
Regressor algorithm is used.

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn import model_selection

# Read data
path_to_train = 'data/train.csv'
originalDF = pd.read_csv(path_to_train)


def roughGBRTest(df):
    """
    Very rough test to see how Gradient Boosting Regressor would perform on
    roughly preprocessed data with only numerical variables that are 
    selected based on subjective intuitions.
    This is just to give us an overview of what to do and set the standard
    benchmark in terms of the performance.
    """
    # rough preprocess
    
    # BEGIN: from https://www.kaggle.com/dansbecker/learning-to-use-xgboost
    # EXPLANATION: The missing values in SalePrice (specified by subset 
    # argument) are inspected and the row (axis = 0) will be dropped if
    # missing. inplace is True so that df is mutated.
    df.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
    # END: from https://www.kaggle.com/dansbecker/learning-to-use-xgboost
    y = df.loc[:, "SalePrice"]
    X = df.loc[:, ["LotArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea", "YrSold"]]
    
    gbr = GradientBoostingRegressor()
    # BEGIN: from https://www.kaggle.com/dansbecker/handling-missing-values
    # EXPLANATION: Imputers can fill in missing values with some numbers. Filling
    # in missing values is usually better than dropping them entirely because
    # those that are not missing might indicate some valuable patterns.
    my_imputer = Imputer()
    X = my_imputer.fit_transform(X)
    # END: from https://www.kaggle.com/dansbecker/handling-missing-values
    cvScores = model_selection.cross_val_score(gbr, X, y, cv=10,
                                               scoring = 'neg_mean_squared_error')
    
    print("Mean Squared Error:", -1 * cvScores.mean())


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
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
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
    originalDF.loc[:, "YearsOld"] = 2008 - originalDF.loc[:, "YearBuilt"]
    
    #create new column called Yard Size (Lot Area - Ground Floor Size)
    lotArea = originalDF.loc[:, "LotArea"]
    gFlrSize = originalDF.loc[:, "GrLivArea"]
    originalDF.loc[:, "YardSize"] = lotArea - gFlrSize
    
    return originalDF


def checkOutliers(originalDF):
    originalDF = featureEngineering(originalDF)
    
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Concatinated dataframes are plotted using 
    #bivariate analysis saleprice/grlivarea
    vsGrLivArea = pd.concat([originalDF.loc[:, 'SalePrice'], originalDF.loc[:, 'GrLivArea']], axis=1)
    vsGrLivArea.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
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
    
    # Identifying the IDs of the two (or four) outliers from GrLivArea
    print(originalDF.sort_values(by = 'GrLivArea', ascending = False)[:4])
    
    # Identifying the IDs of the one outlier from YearsOld
    yearsOldOutlier = (originalDF.loc[:, "YearsOld"] > 120) & (originalDF.loc[:, "SalePrice"] > 400000)
    print(originalDF.loc[yearsOldOutlier])
    
 
def dropOutliers(originalDF):
    # Deleting the outliers
    rowsToDrop = (originalDF.loc[:, "Id"] == 1299) | (originalDF.loc[:, "Id"] == 524) | (originalDF.loc[:, "Id"] == 1183) | (originalDF.loc[:, "Id"] == 692) | (originalDF.loc[:, "Id"] == 186)
                 
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
    all_missing = False
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
    


def testFinalResults(originalDF):

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
    
    print("Decision Tree Regressor")
    print("\tSelected Columns")
    testDTR(selected_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values")
    testDTR(most_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testDTR(corr_dropped_DF)
    
    print()
    
    print("Random Forest Regressor")
    print("\tSelected Columns")
    testRF(selected_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values")
    testRF(most_hot_end_DF)
    print("\tAll Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testRF(corr_dropped_DF)
    
    
def submitResult(originalDF):
    # Read the test data
    testDF = pd.read_csv("data/test.csv")
    ids = testDF.loc[:, "Id"]
    
    # Dropping Outliers first
    originalDF = dropOutliers(originalDF)
    
    y = originalDF.loc[:, "SalePrice"]
    trainDF = originalDF.drop(["SalePrice"], axis=1)

    # Concatinate test and original (= train)
    train_numRow = trainDF.shape[0]
    dataset = pd.concat(objs=[trainDF, testDF], axis=0)
    # One-Hot Encoding to create the full dataset
    dataset = pd.get_dummies(dataset)
    # Splitting them again so that the number of features in
    # the test and original matches
    trainDF = dataset.iloc[:train_numRow, :].copy()
    testDF = dataset.iloc[train_numRow:, :].copy()
    
    # Add Engineered Features
    trainDF = featureEngineering(trainDF)
    # Standardize the Trainingset
    trainDF = standardize(trainDF)
    
    gbr = GradientBoostingRegressor()
    
    trainDF = handleMissingByImputer(trainDF)
    gbr.fit(trainDF, y)

    # Test dataset preprocessing
    testDF = featureEngineering(testDF)
    testDF = standardize(testDF)
    testDF = handleMissingByImputer(testDF)

    predictions = gbr.predict(testDF)
    
    # Create submission file in csv
    submitDF = pd.DataFrame(
                    {
                     "Id": ids,
                     "SalePrice": predictions
                    }
               )
    submitDF.to_csv("data/submission.csv", index = False)
    