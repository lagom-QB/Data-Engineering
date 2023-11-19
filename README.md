# Sports Data Integration and ETL Skill Development

## Project Description

The goal of this project is to develop skills in data integration, ETL (Extract, Transform, Load) processes, and data transformation using a sports dataset. The project will focus on collecting, cleaning, transforming, and loading sports-related data into a target repository for analysis and reporting purposes.
Because the dataset concerns data on the top 5 European leagues and this data seems to be alot at first glance, I'm going to narrow it down by working on data on the *Liga Feminina* for Spain.  
[![StreamLit](https://img.shields.io/badge/StreamLitLink-gray)](https://la-liga.streamlit.app/)

---

## Goal: Predict the resulting scores from a game

The final deliverable of this project is a model which predicts the possible winner of a game and expected results by analysing past matches and performance.  
However, it is worth realising that this is real life and we can't exactly predict results of events such as these.

---

## Project Steps

1. Dataset Selection:  
*The goal of dataset selection is to ensure that the chosen dataset provides the necessary information to answer the research questions or achieve the analysis objectives effectively.*
   - Choose a sports dataset that contains relevant information for analysis. It could include data such as player statistics, match results, team information, or any other sports-related data that is readily available.
   - I'm interested in looking at Women's soccer in Spain

2. Data Collection:  
*Data collection focuses on acquiring the specific data points or variables required for the analysis. It involves designing data collection instruments, such as questionnaires or data collection forms, and implementing them to gather the necessary data. Data collection also includes ensuring data integrity, accuracy, and reliability by following appropriate data collection protocols and techniques.*
   - Identify and collect data from various sources, such as APIs, databases, or web scraping, to build a comprehensive sports dataset. Ensure that the collected data is in a structured format, such as CSV, JSON, or XML.
   - The dataset was taken from [Kaggle](https://datasetsearch.research.google.com/search?src=0&query=dataset%20about%20women%27s%20soccer%20in%20spain&docid=L2cvMTFzZGYweGdxOQ%3D%3D) and is about *Women's soccer*.
   - This dataset includes comprehensive female football-related performance data and player statistics from the top 5 European leagues: Serie A in Italy, Liga Femenina in Spain, Women's Super League in England, Bundesliga Frauen in Germany, and Division 1 Feminin in France. Gathered throughout each season of the respective leagues, the dataset tracks teams, players, matches and a range of important performance metrics. The recently released data provides intriguing insight into team success and player form - covering parameters such as goals scored per game (xGHome), clean sheets (CS), number of opponents' passes allowed (Sweeper_#OPA) as well as individual performance stats such as tackles made per goal kick (Crosses_Stp).

3. Data Cleaning and Preprocessing:
   - Clean the collected data by addressing issues like missing values, inconsistent formatting, and outliers. Standardize the data to ensure consistent data types and formats across the dataset.
   - Focusing on the *matches_checkpoints.csv*, I eliminated a bunch of columns which I couldn't explain and couldn't find a description for from the data source.

4. Feature Engineering:
   - Apply data transformation techniques to enrich the dataset. This involved merging multiple datasets, calculating derived metrics and converting data into a more suitable format for analysis.

5. Model Building:
   _Because there are 2 target variables, I can approach this with Multi-output regression, separate models or sequential models_.  
   - The ML problem is a aregression problem, seeing as the target variables are continous btu independent of each other so I'm going to build separate models for each target.
   - Split the data in train and testing sets.
   - Choose appropriate ML algorithms then train the models on the train data and evaluate models on the testing set using evaluation metrics.
   - Tune the hyperparameters of the model using techniques like gridSerach or randomSearch to optimize their performances.

6. Deployment and Presentation:
   - Prepare the project for deployment by packaging the code, dependencies and documentation.
   - Showcase the project including the problem statement, steps, procedures, results and visualizations.
   - Showcase the key findings, insights and value of the project

---

## Takeaways

- Gained experience in *data integration*, *ETL processes*, and *data transformation* using a sports dataset.
- This enhanced my skills in managing and manipulating data.
- My goal was to challenge myself to build the minimun viable product from a data project, challenging myself to move from the usual jupyter notebook to something useable.

---

## Next steps

- Connect to a realtime database
- Use a bigger dataset not just the female spanish teams.
- Incorporate other statistics into the model such as player statistics.... _This may be heavy and alot to input so maybe automate this??_

### Lessons
```
__Apache Airflow__ supports a few databases:
- SQLite: Lightweight filebased database suitable for small-scale deployments and testing
- PostgreSQL: Relational database widely used in production environments
- MySQL: Popular relational database widely used
- Microsoft SQL Server: Commercial relational database widely used in enterprises
- Oracle: Commercial relational database widely used in enterprises
- Amazon RedShift: Cloud-based data warehouse optimized for analytics workloads
- Google BigQuery: Cloud-based data warehouse optimized for analytics workloads
- Apache Casssandra: Distributed No-SQL database optimized for high scalability and availability
- Apache Hive: Data warehouse infrastructure for data summarization, querying and analytics
```