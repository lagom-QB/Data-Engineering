# Imports
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.discriminant_analysis import StandardScaler

from datetime import datetime

import warnings
from typing import Dict, Any, List, Union
import fire

warnings.filterwarnings('ignore')

import pickle
import os


def tune_model_parameters(model_name: str, X_train: pd.DataFrame, y_train: pd.Series)->Dict[str, Union[str, int, float]]:
    """
    Tune the parameters of the model to get the best performance
    """
    if model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor()
        parameters = {'n_estimators': [50, 100, 150, 200],
                      'learning_rate': [0.01, 0.1, 1, 10, 100]}
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor()
        parameters = {'n_estimators': [50, 100, 150, 200],
                      'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'XGBRegressor':
        model = XGBRegressor()
        parameters = {'n_estimators': [50, 100, 150, 200],
                      'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'LinearRegression':
        model = LinearRegression()
        parameters = {'fit_intercept': [True, False],
                      'positive': [True, False],
                      'n_jobs': [1, 2, 3, 4, 5]}
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
        parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'KNeighborsRegressor':
        model = KNeighborsRegressor()
        parameters = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor()
        parameters = {'n_estimators': [50, 100, 150, 200],
                      'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor()
        parameters = {
            'objective':['regression'],
            'num_leaves':[5, 10, 15, 20, 25, 30],
            'learning_rate':[0.01, 0.05, 0.1, 0.5, 1],
            'n_estimators':[100, 200, 300, 400, 500],
            'max_bin':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'bagging_fraction':[0.5, 0.75, 1],
            'bagging_freq':[5, 10, 15, 20],
            'feature_fraction':[0.5, 0.75, 1],
            'feature_fraction_seed':[5, 10, 15, 20],
            'force_row_wise':['False'],
            'force_col_wise':['False'],
            'bagging_seed':[5, 10, 15, 20],
            'min_data_in_leaf':[5, 10, 15, 20],
            'min_sum_hessian_in_leaf':[5, 10, 15, 20],
            'verbosity':[-1]
        }
    elif model_name == 'CatBoostRegressor':
        model = CatBoostRegressor()
        parameters = {'n_estimators': [50, 100, 15, 20],
                      'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    elif model_name == 'SVR':
        model = SVR()
        parameters = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf', 'poly', 'sigmoid']}
    elif model_name == 'Ridge':
        model = Ridge()
        parameters = {'alpha': [1, 10, 100, 1000],
                      'fit_intercept': [True, False],
                      'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    elif model_name == 'Lasso':
        model = Lasso()
        parameters = {'alpha': [1, 10, 100, 1000],
                      'fit_intercept': [True, False]}
    elif model_name == 'ElasticNet':
        model = ElasticNet()
        parameters = {'alpha': [1, 10, 100, 1000],
                      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                      'fit_intercept': [True, False]}
    elif model_name == 'SGDRegressor':
        model = SGDRegressor()
        parameters = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'penalty': ['l2', 'l1', 'elasticnet'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                      'fit_intercept': [True, False],
                      'shuffle': [True, False],
                      'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']}
    elif model_name == 'KernelRidge':
        model = KernelRidge()
        parameters = {'alpha': [1, 10, 100, 1000],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'degree': [1, 2, 3, 4, 5, 6],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    else:
        raise ValueError('Model not supported yet')
    
    # Tune the model
    # tuner = GridSearchCV(model, parameters, scoring='r2_score', cv=10) # type: ignore
    tuner = RandomizedSearchCV(model, parameters, n_iter=500 , scoring='neg_mean_absolute_error', n_jobs=-1, cv=10) # type: ignore
    tuner.fit(X_train, y_train)


    # Get the best parameters
    best_params = tuner.best_params_
    best_score = tuner.best_score_
    best_estimator = tuner.best_estimator_

    return [model,best_params] # type: ignore

n_iterations = 50
    
def train_model(rankings: Dict[str, Dict[str, Union[str,int, float]]], model_name: str, features: pd.DataFrame, target: pd.Series) -> Dict[str, Union[str,int, float]]:
    """
        Tunes and trains a model and returns the evaluation metrics
    """
    start_time = datetime.now()
    # Split the data in train and test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model, params = tune_model_parameters(model_name, X_train, y_train) # type: ignore
    # print(f"\n\n{model} Best parameters: {params}")

    model.set_params(**params) # type: ignore # Assign the best parameters to the model
    mae_train = []

    for epoch in range(n_iterations):
        model.fit(X_train, y_train) # type: ignore
        
        y_pred = model.predict(X_train) # type: ignore
        
        curr_mae_train = mean_absolute_error(y_train, y_pred)
        mae_train.append(curr_mae_train)
        
    plt.figure(figsize=(14,6))
    
    plt.xlim(0, n_iterations)
    plt.ylim(-1, 1)
    
    plt.plot(mae_train, linewidth=.8, label='Training MSE')
    
    plt.legend()
    plt.xlabel(f'mse train')
    plt.title(f'Model: {model.__class__.__name__} | Epoch: {epoch+1}/{n_iterations}\n Training MSE: {curr_mae_train:.4f}') # type: ignore
    # plt.pause(1.2)
    
    plt.show()
    
    model.fit(X_train, y_train) # type: ignore
    y_pred = model.predict(X_test) # type: ignore # Predict the target

    mae = min(min(mae_train), mean_absolute_error(y_test, y_pred))

    end_time = datetime.now()
    runtime = end_time - start_time
    # print(f"{model}\nMAE {model_name}: {mae} \nRuntime: {runtime}\n====================\n")
    # Create a dictionary to store the model name, its params, and its mae so we can compare them later
    
    rankings[model_name] = {}
    rankings[model_name]['model'] = model
    rankings[model_name]['params'] = params
    rankings[model_name]['mae'] = mae # type: ignore 
    rankings[model_name]['time'] = end_time - start_time # type: ignore

    return rankings # type: ignore

def training_process(ranking_score: Dict[str, Dict[str, Union[str, int, float]]], features: pd.DataFrame, target: pd.Series) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
        Trains a model and returns the ranked models
    """
    train_model(ranking_score,'LinearRegression', features, target)
    train_model(ranking_score,'AdaBoostRegressor', features, target)
    train_model(ranking_score,'RandomForestRegressor', features, target)
    train_model(ranking_score,'XGBRegressor', features, target)
    train_model(ranking_score,'DecisionTreeRegressor', features, target)
    train_model(ranking_score,'KNeighborsRegressor', features, target)
    train_model(ranking_score,'GradientBoostingRegressor', features, target)
    train_model(ranking_score,'Ridge', features, target)
    train_model(ranking_score,'Lasso', features, target)
    train_model(ranking_score,'ElasticNet', features, target)
    train_model(ranking_score,'SGDRegressor', features, target)
    train_model(ranking_score,'KernelRidge', features, target)

    return ranking_score

def save_best_model(name: str, model_list: Dict[str, Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
    """
        Saves the best model and its info to a pickle file
        TODO::
            - Save all the models to a database so we can compare them later
    """

    model_comparisons = pd.DataFrame.from_dict(model_list, orient='index')
    model_comparisons.index.name = 'model'
    
    # Sort the models by their mae
    model_comparisons = model_comparisons.sort_values(by='mae').reset_index(drop=True)
    # print(model_comparisons)

    # Save th best model to a pickle file from the model_comparisons dataframe
    best_model = model_comparisons.loc[0, 'model']
    best_model_params = model_comparisons.loc[0, 'params']
    best_model_info = {'model': best_model, 'params': best_model_params}

    # if file path exists, remove all files in it
    model_dir = 'airflow/dags/model_dir'
    os.makedirs(model_dir, exist_ok=True)

    # Remove all files in the model_dir ending with {name}.pkl
    for file in os.listdir(model_dir):
        if file.endswith(f'{name}.pkl'):
            os.remove(f'{model_dir}/{file}')

    # Save the best model to a pickle file
    with open(f'{model_dir}/{best_model.__class__.__name__}_{name}.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    # Return the best model from model_comparison with the lowest mae
    return model_comparisons 

if __name__ == '__main__':
    fire.Fire(training_process)
    fire.Fire(save_best_model)