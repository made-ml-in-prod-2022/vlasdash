from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable

RAW_DATA_DIR = "/data/raw/{{ ds }}"
TRANSFORMER_DIR = "/data/models/{{ ds }}"
PREDICTS_DIR = "/data/predictions/{{ ds }}"
MODEL_DIR = Variable.get('model_dir')

default_args = {
    "owner": "vlasdash",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "predict_pipeline",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command=f"--data-dir {RAW_DATA_DIR} --model-dir {MODEL_DIR} --transformer-dir {TRANSFORMER_DIR} --predicts-dir {PREDICTS_DIR}",
        network_mode="bridge",
        task_id="predict",
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
