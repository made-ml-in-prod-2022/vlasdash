# Homework3

## Запуск
~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
~~~

Для дага predict_pipeline необходимо добавить aiflow variable по ключу model_dir со значением /data/models/<дата в формате YYYY-MM-DD> 
