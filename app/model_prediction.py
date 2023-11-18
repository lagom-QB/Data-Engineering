# Imports
import warnings
import fire

warnings.filterwarnings('ignore')

import pickle
import os

def predict_res(features: list) -> str:
    # Get the model in the model_dir which ends in model_type
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

    # print(model_home, model_away, sep='\n\n')

    prediction_home = model_home.predict(features)
    prediction_away = model_away.predict(features)

    return (f'predicting ... {abs(int(prediction_home))} : {abs(int(prediction_away))}')

if __name__ == '__main__':
    fire.Fire(predict_res)