# Imports
import pandas as pd

from typing import Optional
import fire

file_loc       = 'airflow/dags/assets/matches-checkpoint.csv'

def load_data_from_source(file_location: Optional[str] = file_loc) -> pd.DataFrame:
    """
    Loads the data from the source file and returns a pandas dataframe.
    """
    df = pd.read_csv(file_location)

    print(df.shape, df.info())

    return df

if __name__ == '__main__':
    fire.Fire(load_data_from_source)