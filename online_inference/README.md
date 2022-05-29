# Homework 2
## Запуск локально с помощью Docker
~~~
docker build -t whoemta/hw2:v2 .
docker run -p 8000:8000 whoemta/hw2:v2
~~~

## Запуск из DockerHub
~~~
docker pull whoemta/hw2:v2
docker run -p 8000:8000 whoemta/hw2:v2
~~~

## Запуск тестов
~~~
pytest test/test.py
~~~

## Запуска скрипта с запросами
"data/heart_cleveland_upload.csv" - путь до данных, из которых формируется запрос
~~~
python request.py "data/heart_cleveland_upload.csv"
~~~

## Структура проекта

    ├── README.md             <- The top-level README for developers using this project.
    ├── data
    │   └── heart_cleveland_upload.csv   <- The data for request.
    │
    ├── models                <- Trained and serialized models and transformer
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment
    │
    │
    ├── entities          <- BaseModels                   
    │   │
    │   ├── condition_response.py
    │   │
    │   └── heart_disease_request.py  
    │
    ├── test          <- test
    │   │
    │   └── test.py  <- Test for /predict
    │  
    ├── app.py       <- Online inference for model
    │
    ├── Dockerfile
    │
    └── request.py <- A script that makes requests to the app

--------
