from datetime import datetime, timedelta
import time

from airflow import DAG
from airflow.operators.python import PythonOperator # for python commands

with DAG(
    dag_id="trial_dag",
    schedule="* * * * * ", # every minute
    start_date=datetime(2023, 10, 29),
    catchup=False,
    description="Trial DAG",
    tags=["trial-run", "bash", "python"],
    default_args={
        "owner": "Brenda",
        "depends_on_past": True,
        "retries": 2,
        "retry_delay": timedelta(minutes=5)
    }
) as dag:
    # Current datetime
    task1 = PythonOperator(
        task_id="print_current_date",
        python_callable=lambda: (datetime.now())
    )
    task1.doc_md = """
    #### Task1 Documentation
        Task1 prints the current date
    """

    task2 = PythonOperator(
        task_id="hold_5_seconds",
        depends_on_past=True,
        python_callable=lambda: time.sleep(5),
        retries=2
    )
    task2.doc_md = """
    #### Task2 Documentation
        Task2 sleeps for 5 seconds
    """

    dag.doc_md = """
    ### DAG Documentation
        This DAG is for trial purpose to experiment with airflow and bash commands
    """

    task1 >> task2 # task1 must be completed before task2 can be started