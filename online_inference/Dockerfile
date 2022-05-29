FROM python:3.8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD models models
ADD entities entities
COPY app.py app.py

WORKDIR .

ENV PATH_TO_MODEL="models/model.pkl"
ENV PATH_TO_TRANSFORMER="models/transformer.pkl"

CMD python ./app.py
