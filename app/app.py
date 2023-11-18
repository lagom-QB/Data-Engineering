import streamlit as st
# Imports
import pandas as pd

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

import pickle



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