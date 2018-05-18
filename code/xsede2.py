"""
    Created on Mon Apr 16 16:04:30 2018
    
    Data Mining Final Project
    
    @author: Shuto Araki and Taras Tataryn
    
    @dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    
    This file is for hyperparameter tuning in XGBoost.
    The computation requires about 30 minutes using
    the Virtual Machine with 120GB of memory and 44 CPUs.
    
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from xgboost import XGBRegressor


# Read data
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
    
    return originalDF


def dropOutliers(originalDF):
    # Deleting the outliers
    rowsToDrop = (originalDF.loc[:, "Id"] == 1299) | (originalDF.loc[:, "Id"] == 524) | (originalDF.loc[:, "Id"] == 1183) | (originalDF.loc[:, "Id"] == 692) | (originalDF.loc[:, "Id"] == 186)
    
    originalDF = originalDF.drop(originalDF[rowsToDrop].index)
    
    return originalDF


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

# Tuning the GradientBoostingRegressor with GridSearchCV
def tuneGBR(df):
    X, y = preprocess(df)
    param_test1 = {'n_estimators':range(100,500,10)}
    gsearch1 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.1,
                min_samples_split = 10,
                min_samples_leaf = 1,
                max_depth = 4,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 1
                ), 
            param_grid = param_test1, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch1.fit(X, y)
    print("Best n_estimators and its score", gsearch1.best_params_, gsearch1.best_score_, sep='\n')
    

def tunedGBR(X, y):
    gbr = GradientBoostingRegressor(learning_rate = 0.1, 
                                    n_estimators=360, 
                                    min_samples_split = 10, 
                                    min_samples_leaf = 1, 
                                    max_depth=4, 
                                    max_features='sqrt',
                                    subsample = 0.8,
                                    random_state=0)
    cvScores = model_selection.cross_val_score(gbr, X, y, cv=10,
                                               scoring = 'neg_mean_squared_error')
    print("\tMean Squared Error:", np.sqrt(-1 * cvScores.mean()))


def testResultsModels(originalDF):
    X, y = preprocess(originalDF)
    print("Graident Boosting Regressor: n_estimators = 360")
    tunedGBR(X, y)
    


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


def model_test_xgb(df, n_jobs):
    X, y = preprocess(df)
    xgb = XGBRegressor(random_state=0)
    param_grid = {
        'n_estimators': [1000],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'min_child_weight': range(1, 6),
        'max_depth': range(2, 7),
        'learning_rate': [0.01],
        'subsample': [0.75, 0.8, 0.85]
    }
    model = model_selection.GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=n_jobs, cv=10, scoring=RMSE)
    model.fit(X, y)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
# Using the xxlarge jetstream image (44 vCPUs)
#model_test_xgb(originalDF, n_jobs=44)

def model_test_xgb_regulation(df, n_jobs):
    X, y = preprocess(df)
    xgb = XGBRegressor(n_estimators=1000,
                       learning_rate=0.01, 
                       min_child_weight=3,
                       max_depth=6, 
                       gamma=0, 
                       subsample=0.8, 
                       random_state=0)
    param_grid = {
            'colsample_bytree': [i/100.0 for i in range(75,90,5)],
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    model = model_selection.GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=n_jobs, cv=10, scoring=RMSE)
    model.fit(X, y)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

model_test_xgb_regulation(originalDF, n_jobs=44)