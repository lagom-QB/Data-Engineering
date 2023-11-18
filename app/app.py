from unittest import result
import streamlit as st
# Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from datetime import datetime, timedelta

# import sqlite3
# from airflow import DAG
# from airflow.utils.dates import days_ago
# from airflow.operators.python_operator import PythonOperator

import nbformat

import warnings
from typing import Optional, Tuple, Callable, Dict, Any, List, Union
import fire

warnings.filterwarnings('ignore')

# Create a requirements.txt file with the necessary packages
# !pip freeze > airflow/dags/requirements.txt

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.discriminant_analysis import StandardScaler

from fastapi import FastAPI

import pickle
import os
from load_data import load_data_from_source
from transform_data import transform_data_into_features_and_targets
from baseline_model import train_baseline

from train_models import train_model, save_best_model, training_process

from model_prediction import predict_res



spanish_squads = ['Sevilla', 'Sporting Huelva', 'Athletic Club', 'Levante Planas',
                  'UDG Tenerife', 'Villarreal', 'Madrid CFF', 'Barcelona',
                  'Atlético Madrid', 'Real Madrid', 'Alhama', 'Alavés',
                  'Real Sociedad', 'Levante', 'Real Betis', 'Valencia']

def prediction(data):
    model_home = pickle.load(open('airflow/dags/model_dir/model_home.pkl', 'rb'))
    model_away = pickle.load(open('airflow/dags/model_dir/model_away.pkl', 'rb'))
    return model_home.predict(data), model_away.predict(data)

st.title('La Liga Score Predictor')
st.markdown('Model to predict the scores of La Liga matches and the subsequent match winner')
st.divider()

st.header('Match Predictor')

day = st.date_input('Day', value=datetime.now())

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('Teams')
    st.text('Home team')
    home_team = st.selectbox('Select home team', spanish_squads, label_visibility='hidden')
    st.text('Away team')
    away_team = st.selectbox('Select away team', spanish_squads, label_visibility='hidden')
with col3:
    st.subheader('Match details')
    st.text('Week')
    wk = st.number_input('Select week', min_value=1, max_value=38, value=1, label_visibility='hidden')
    st.text('Time')
    time = st.time_input('Select time', value=datetime.now(), label_visibility='hidden')

st.divider()

if st.button('Predict'):
    st.write(f'Predicting ... {home_team} vs {away_team}')
    # save the inputs into a dataframe
    match = pd.DataFrame({'Week': [wk], 'Day': [day], 'Time': [time], 'Home': [home_team], 'Away': [away_team]})
    # run the prediction models
    result = prediction(match)
    st.write(result)

st.markdown(
    '`Created by` [Brenda](https://github.com/lagom-QB) | \
         `Code:` [GitHub](https://github.com/lagom-QB/Data-Engineering)')