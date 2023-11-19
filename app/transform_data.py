# Imports
import pandas as pd

from typing import Optional, Tuple
import fire

import warnings

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, PolynomialFeatures
warnings.filterwarnings('ignore')

def encode_categorical_varaibles(df: pd.DataFrame)-> pd.DataFrame:
    """
    Encodes the categorical variables in the dataframe
    :param df: Dataframe with the data
    :return: Dataframe with the categorical variables encoded
    """
    # Get the categorical columns
    categorical_vars = df.select_dtypes(include=['object']).columns
    spanish_teams = ['Alavés', 'Alhama', 'Athletic Club', 'Atlético Madrid',
                                'Barcelona', 'Levante', 'Levante Planas', 'Madrid CFF',
                                'Real Betis', 'Real Madrid', 'Real Sociedad', 'Sevilla',
                                'Sporting Huelva', 'UDG Tenerife', 'Valencia', 'Villarreal']

    df['Numeric_Day']=df['Day'].apply(lambda x: 2 if x == 'Tue' else 3 if x == 'Wed' else 4 if x == 'Thu' else 6 if x == 'Sat' else 7) # Encode the Day column
    
    encoder = LabelEncoder() # Initialize the encoder
    teams = encoder.fit_transform(spanish_teams) # Encode the teams
    df['Numeric_Home'] = encoder.transform(df['Home']) # Encode the Home column
    df['Numeric_Away'] = encoder.transform(df['Away']) # Encode the Away column

    encoder_time = OrdinalEncoder() # Initialize the encoder
    df['Numeric_Time'] = encoder_time.fit_transform(df[['Time']]) # Encode the Time column
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0) # Fill the NaN values with 0
    interaction_terms = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_var   = interaction_terms.fit_transform(df[['xGHome', 'xGAway']])


    return pd.DataFrame(interaction_var, columns=['xGHome_xGAway_1', 'xGHome_xGAway_2', 'xGHome_xGAway_3'])

def transform_data_into_features_and_targets(
        # Dataframe with the data
        df: pd.DataFrame,
        score: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms the data into features and targets for either the home or away team.
    :param df: Dataframe with the data
    :param score: Score to be used as the targets
    :return: Tuple with the features and targets
    """
    df = encode_categorical_varaibles(df) # Encode the categorical variables
    iv = create_interaction_features(df) # Create the interaction features
    
    df = pd.concat([df, iv], axis=1) # Concatenate the interaction features with the dataframe

    targets = ['ScoreHome', 'ScoreAway'] # The targets are the score variables
    target  = None # Initialize the target variable
    
    if score == 'Home':
        target = targets[0]
    elif score == 'Away':
        target = targets[1]
    else:
        targets = [] # If the score is empty, the targets are empty

    y = None # Initialize the targets variable
    if len(targets)>0:
        # Create the targets from the score
        y = df[target].copy()
    
    # Create the features
    # The columns that contain 'id' are not useful for the model so I drop them
    for col in df.columns:
        if col.lower().__contains__('id') or df[col].dtype == 'object':
            df = df.drop(col, axis=1)
    X = df.drop(columns=targets)
    
    # if score is empty, return just the features
    if score == '':
        print(f'Returning just the features .... {X}')
        return X # type: ignore
    else:
        return X, y

if __name__ == '__main__':
    fire.Fire(transform_data_into_features_and_targets)