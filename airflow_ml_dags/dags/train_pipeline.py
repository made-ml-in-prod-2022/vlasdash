from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

RAW_DATA_DIR = "/data/raw/{{ ds }}"
SPLIT_DATA_DIR = "/data/splitted/{{ ds }}"
PROCESS_DATA_DIR = "/data/processed/{{ ds }}"
TRANSFORMER_DIR = "/data/models/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
METRICS_DIR = "/data/metrics/{{ ds }}"

default_args = {
    "owner": "vlasdash",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(1),
) as dag:
    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {RAW_DATA_DIR} --output-dir {SPLIT_DATA_DIR}",
        network_mode="bridge",
        task_id="split_data",
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

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {SPLIT_DATA_DIR} --output-dir {PROCESS_DATA_DIR} --transformer-dir {TRANSFORMER_DIR}",
        network_mode="bridge",
        task_id="preprocess_data",
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

    train = DockerOperator(
        image="airflow-train",
        command=f"--data-dir {PROCESS_DATA_DIR} --model-dir {MODEL_DIR}",
        network_mode="bridge",
        task_id="train_model",
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

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-dir {SPLIT_DATA_DIR} --transformer-dir {TRANSFORMER_DIR} --model-dir {MODEL_DIR} --metrics-dir {METRICS_DIR}",
        network_mode="bridge",
        task_id="validate_model",
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

    split >> preprocess >> train >> validate
