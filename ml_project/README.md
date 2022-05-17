Homework 1
==============================

Структура проекта
------------

    ├── README.md             <- The top-level README for developers using this project.
    ├── data
    │   └── raw               <- The original, immutable data dump.
    │
    ├── models                <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py              <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── src                   <- Source code for use in this project.
        ├── __init__.py       <- Makes src a Python module
        │
        ├── data              <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── entities          <- Dataclasses
        │    │
        │    ├── feature_params.py
        │    │
        │    ├── metric_params.py
        │    │	      
        │    ├── model_forest_params.py
        │    │
        │    ├── model_sgd_params.py
        │    │
        │    ├── predict_pipeline_params.py
        │    │
        │    ├── splitting_params.py
        │    │
        │    ├── train_pipeline_params.py
        │    │
        │    └── training_params.py
        │
        ├── features          <- Scripts to turn raw data into features for modeling
        │    │
        │    └── build_features.py
        │
        ├── models            <- Scripts to train models and then use trained models to make
        │    │                   predictions
        │    ├── predict_model.py
        │    └── train_model.py
        │
        ├ tests               <- Scripts to test code 
        │
        │
        ├ train_pipeline.py   <- Scripts to train pipeline
        │
        ├ predict_pipeline.py <- Scripts to predict pipeline
        │
        └ run_tests.py        <- Scripts to run all tests


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
