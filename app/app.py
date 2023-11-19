import streamlit as st
# Imports
import pandas as pd

from datetime import datetime

import warnings

from transform_data import transform_data_into_features_and_targets

warnings.filterwarnings('ignore')

import pickle
import os



spanish_squads = ['Sevilla', 'Sporting Huelva', 'Athletic Club', 'Levante Planas',
                  'UDG Tenerife', 'Villarreal', 'Madrid CFF', 'Barcelona',
                  'Atlético Madrid', 'Real Madrid', 'Alhama', 'Alavés',
                  'Real Sociedad', 'Levante', 'Real Betis', 'Valencia']

def prediction(data):
    #  Load the modesl. The model_home is the model which ends in home.pkl and model_away is the model which ends in away.pkl
    model_dir = 'airflow/dags/model_dir/'
    model_home_loc = [model_dir+file for file in os.listdir(model_dir) if file.endswith('home.pkl')][0]
    model_away_loc = [model_dir+file for file in os.listdir(model_dir) if file.endswith('away.pkl')][0]

    try:
        with open(model_home_loc, 'rb') as file:
            model_home = pickle.load(file)
    except FileNotFoundError:
        return 'Model file not found for home'
    
    try:
        with open(model_away_loc, 'rb') as file:
            model_away = pickle.load(file)
    except FileNotFoundError:
        return 'Model file not found for away'
    
    if not hasattr(model_home, 'predict') or not hasattr(model_away, 'predict'):
        return 'Invalid model object'
    
    # Transform the data into features
    features = transform_data_into_features_and_targets(data, '')
    # st.write(f'features... \n{features.iloc[0]}')

    # Run the prediction models
    prediction_home = model_home.predict(features)
    prediction_away = model_away.predict(features)

    return st.markdown(f'`{abs(int(prediction_home))}` : `{abs(int(prediction_away))}`')

st.title('La Liga Score Predictor')
st.markdown('Model to predict the scores of La Liga matches and the subsequent match winner')
st.divider()

st.header('Match Predictor')

day = st.date_input('Day', value=datetime.now())

col1, col2, col3, col4 = st.columns(4)
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
with col4:
    st.subheader('Expectations')
    st.text('Expected goals Home')
    xGHome = st.number_input('Home goals', min_value=0, max_value=5, value=0, step=1, label_visibility='hidden')
    st.text('Expected goals Away')
    xGAway = st.number_input('Away goals', min_value=0, max_value=5, value=0, step=1, label_visibility='hidden')

st.divider()

if st.button('Predict'):
    # save the inputs into a dataframe
    match = pd.DataFrame({'Week': [wk], 'Day': [day], 'Time': [time], 'Home': [home_team], 'Away': [away_team], 'xGHome': [xGHome], 'xGAway': [xGAway]})
    st.write(match)

    st.markdown(f'`{home_team}` vs `{away_team}`')
    # run the prediction models
    prediction(match)
    

st.markdown(
    '`Created by` [Brenda](https://github.com/lagom-QB) | \
         `Code:` [GitHub](https://github.com/lagom-QB/Data-Engineering)')