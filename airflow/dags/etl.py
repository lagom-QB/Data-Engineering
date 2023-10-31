from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from datetime import datetime, timedelta
import logging


# Define the DAG so it runs on an hourly basis
with DAG(
    dag_id="etl", 
    schedule_interval="* * * * *", # minute hour day month day_of_week
    catchup=False,
    description='Load data into an SQLite databse through airflow',
    start_date=datetime.now(),
    default_args={
        'owner': 'Brenda',
        'depends_on_past': True,
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    },  # DAG arguments
    tags=['etl', 'airflow'],
) as dag:
    # ----- Variables -----
    file_location  = 'assets/matches-checkpoint.csv'
    useless_ids    = ['Away_id','Home_id','Match_id','League_id']
    spanish_squads = ['Sevilla', 'Sporting Huelva', 'Athletic Club', 'Levante Planas',
                      'UDG Tenerife', 'Villarreal', 'Madrid CFF', 'Barcelona',
                      'Atlético Madrid', 'Real Madrid', 'Alhama', 'Alavés',
                      'Real Sociedad', 'Levante', 'Real Betis', 'Valencia']
    # ---------------------
    # Extract the data from the csv file
    def extract_data():
        import pandas as pd
        import logging

        matches = pd.read_csv(file_location) # read the csv file
        logging.info(f"Dataframe shape: {matches.shape} \n {matches.head()}")

        matches = matches[(matches['Home'].isin(spanish_squads)) | (matches['Away'].isin(spanish_squads))] # filter out the spanish teams 
        matches = matches.drop(useless_ids, axis=1).reset_index(drop=True) # drop the useless ids
        matches['Date'] = pd.to_datetime(matches['Date']) # convert the date column to datetime

        logging.info(matches.head())

        return matches
    extract_data.doc_md = """
        file_loc = file location from where the data is to be extracted
        useless_ids = list of ids that are not required
        spanish_squads = list of spanish female teams
        TODO:: Extract the data from the csv file
                          """
    extracted_data = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
        op_kwargs={
            "file_loc": file_location,
            "useless_ids": useless_ids,
            "spanish_squads": spanish_squads,
        },
    )
    # ---------------------
    # Transform the data
    def transform_data(matches):
        import pandas as pd
        import logging

        matches = matches.dropna() # drop the null values
        matches['Date'] = pd.to_datetime(matches['Date'])
        matches['Time'] = pd.to_datetime(matches['Time'], format='%H:%M:%S').dt.time # convert the time column to time

        matches['GoalDifference'] = matches['ScoreHome'] - matches['ScoreAway']
    
        # Result: You can calculate the result of each match by comparing the "ScoreHome" and "ScoreAway" columns. If the home team scored more goals than the away team, then the home team won the match. If the home team scored fewer goals than the away team, then the home team lost the match. If both teams scored the same number of goals, then the match was a draw.
        matches['Result'] = matches['Score'].apply(lambda x: 'Win' if x[0] > x[2] else 'Draw' if x[0] == x[2] else 'Loss')

        # Expected Goals Difference: Similar to the goal difference, you can calculate the expected goals difference by subtracting the "xGAway" from the "xGHome" column. This metric represents the difference in expected goals between the home and away teams in each match.
        matches['ExpectedGoalDifference'] = matches['xGHome'] - matches['xGAway']

        # Points: You can calculate the points earned by each team using a scoring system (e.g., 3 points for a win, 1 point for a draw, and 0 points for a loss). You can create a new column called "Points" and assign the corresponding points based on the match result in the "Score" column.
        matches['Points'] = matches['Score'].apply(lambda x: 3 if x[0] > x[2] else 1 if x[0] == x[2] else 0)

        # Expected Points: Similar to the points metric, you can calculate the expected points earned by each team using a similar scoring system but based on the expected goals (e.g., 3 points for xGHome > xGAway, 1 point for xGHome = xGAway, and 0 points for xGHome < xGAway). You can create a new column called "ExpectedPoints" and assign the corresponding expected points based on the expected goals in the "xGHome" and "xGAway" columns.
        matches['ExpectedPoints'] = matches['Score'].apply(lambda x: 3 if x[0] > x[2] else 1 if x[0] == x[2] else 0)

        # Win Percentage: You can calculate the win percentage for each team by dividing the number of wins (based on the "Score" column) by the total number of matches played.
        wins = matches[matches['Result'] == 'Win'].groupby('Home').size()
        total_matches = matches.groupby('Home').size()
        win_percentage = (wins / total_matches) * 100
        # Add win percentage to matches dataframe
        matches['WinPercentage'] = matches['Home'].map(win_percentage)

        matches['TotalGoals'] = matches['ScoreHome'] + matches['ScoreAway']

        matches['xGRatio'] = matches['xGHome'] / (matches['xGHome'] + matches['xGAway'])

        def get_points(row):
            if row['Result'] == 'Win':
                return 3
            elif row['Result'] == 'Draw':
                return 1
            else:
                return 0
        matches['Points'] = matches.apply(get_points, axis=1)

        logging.info(f"Dataframe transformed: {matches.head()}")

        return matches
    transform_data.doc_md = """
    TODO::
        - Convert data types
        - Calculate and derive metrics
        - Filter data and generate statistics
                            """
    transformed_data = PythonOperator(task_id="transform_data", 
                                      python_callable=transform_data,
                                      op_kwargs={'matches': extracted_data}
                                     )
    # ---------------------
    # Load the data into the database
    def load_data(matches):
        import sqlite3
        import logging

        # Connect to database
        conn = sqlite3.connect('assets/spanish_matches.db')

        # Create cursor
        c = conn.cursor()

        c.execute("""CREATE TABLE IF NOT EXISTS matches (
            Wk INTERGER,
            Day TEXT,
            Date DATE,
            Time TIME,
            Home TEXT,
            xGHome FLOAT,
            Score TEXT,
            xGAway FLOAT,
            Away TEXT,
            xPHome FLOAT,
            xPAway FLOAT,
            ScoreHome INTERGER,
            ScoreAway INTERGER,
            GoalDifference INTERGER,
            Result TEXT,
            ExpectedGoalDifference FLOAT,
            Points INTERGER,
            ExpectedPoints INTERGER,
            WinPercentage FLOAT,
            TotalGoals INTERGER,
            xGRatio FLOAT
        )""") # Create the matches table

        logging.info(c.execure("SELECT * FROM matches").fetchall()) # Print the matches table

        matches.to_sql('matches', conn, if_exists='replace', index=False) # Insert the values from the matches dataframe into the matches table

        conn.commit() # Commit changes

        c.close() # Close cursor
        conn.close() # Close connection
    load_data.doc_md = """
    TODO::
        Load the data into the database after creating the table
    """
    loaded_data = PythonOperator(task_id="load_data", 
                                 python_callable=load_data,
                                 op_kwargs={'matches': transformed_data})
    # ---------------------
    # Validate the data
    # ---------------------

    # Define the order of the tasks
    extracted_data >> transformed_data >> loaded_data
