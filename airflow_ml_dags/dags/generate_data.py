from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

RAW_DATA_DIR = "/data/raw/{{ ds }}"
DATA_SIZE = 100

default_args = {
    "owner": "vlasdash",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command=f"--output-dir {RAW_DATA_DIR} --data-size {DATA_SIZE}",
        network_mode="bridge",
        task_id="generate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/home/dasha/технопарк/MLProduction/vlasdash/airflow_ml_dags/data",
                target="/data",
                type='bind'
            )
        ]
    )
