# Imports
import pandas as pd

from typing import Optional
import fire

file_loc       = 'airflow/dags/assets/matches-checkpoint.csv'
spanish_squads = ['Sevilla', 'Sporting Huelva', 'Athletic Club', 'Levante Planas',
                  'UDG Tenerife', 'Villarreal', 'Madrid CFF', 'Barcelona',
                  'Atlético Madrid', 'Real Madrid', 'Alhama', 'Alavés',
                  'Real Sociedad', 'Levante', 'Real Betis', 'Valencia']

def load_data_from_source(file_location: Optional[str] = file_loc) -> pd.DataFrame:
    """
    Loads the data from the source file and returns a pandas dataframe.
    """

    df = pd.read_csv(file_location) # type: ignore
    # Remove the Home and Away teams that are not in the spanish_squads list
    df = df[(df['Home'].isin(spanish_squads)) | (df['Away'].isin(spanish_squads))]

    print(df.shape, df.info())

    # Drop the columns that are not needed or would not be available for prediction
    useless = ['xPHome', 'xPAway', 'Home_id','Away_id','Match_id','League_id'] 
    df = df.drop(columns=useless)

    return df

if __name__ == '__main__':
    fire.Fire(load_data_from_source)