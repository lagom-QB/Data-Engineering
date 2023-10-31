# Project Title: Sports Data Integration and ETL Skill Development

## Project Description

The goal of this project is to develop skills in data integration, ETL (Extract, Transform, Load) processes, and data transformation using a sports dataset. The project will focus on collecting, cleaning, transforming, and loading sports-related data into a target repository for analysis and reporting purposes.
Because the dataset concerns data on the top 5 European leagues and this data seems to be alot at first glance, I'm going to narrow it down by working on data on the *Liga Feminina* for Spain.

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
   - Focusing on the *all_players.csv* and the *matches_checkpoints.csv*, I eliminated a buch of columns which I couldn't explain and couldn't find a description for on the data source.

4. Data Transformation:
   - Apply data transformation techniques to enrich the dataset. This could involve merging multiple datasets, calculating derived metrics (e.g., batting average, win percentage), or converting data into a more suitable format for analysis.

5. ETL Process Design:
   - Design an ETL process to automate the extraction, transformation, and loading of the sports data. Define the workflow and dependencies between various steps, ensuring that the process is efficient, scalable, and reliable.

6. Data Integration:
   - Integrate the cleaned and transformed sports data into a target repository such as a relational database, data warehouse, or cloud-based storage. Ensure that the data is structured and organized for efficient querying and analysis.

7. Validation and Quality Assurance:
   - Implement data validation checks to ensure the accuracy and quality of the integrated data. Perform data profiling, data integrity checks, and cross-checks against external sources to validate the data.

8. Reporting and Analysis:
   - Utilize data visualization tools or programming libraries to generate meaningful insights and reports from the integrated dataset.

9. Documentation and Presentation:
   - Document each step of the project, including the data sources, data transformations, and ETL processes. Create a presentation summarizing the project's objectives, methodology, findings, and potential use cases.

---

## Takeaways

- Gained experience in *data integration*, *ETL processes*, and *data transformation* using a sports dataset.
- This enhanced my skills in managing and manipulating data

### Lessons

__Apache Airflow__ supports a few databases:

- SQLite: *Lightweight filebased database suitable for small-scale deployments and testing*

- PostgreSQL *Relational database widely used in production environments*
- MySQL *Popular relational database widely used*
- Microsoft SQL Server *Commercial relational database widely used in enterprises*
- Oracle *Commercial relational database widely used in enterprises*
- Amazon RedShift *Cloud-based data warehouse optimized for analytics workloads*
- Google BigQuery *Cloud-based data warehouse optimized for analytics workloads*
- Apache Casssandra *Distributed No-SQL database optimized for high scalability and availability*
- Apache Hive *Data warehouse infrastructure for data summarization, querying and analytics*
