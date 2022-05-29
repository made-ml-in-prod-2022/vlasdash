import sys
import logging
import requests
import click
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)


@click.command(name="send_request")
@click.argument("path_to_data")
def send_request(path_to_data: str):
    data = pd.read_csv(path_to_data)
    length = len(data)

    for i in range(length):
        request = data.iloc[i].to_dict()
        request["id"] = i
        logger.info(f"Request data: {request}")
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json=[request],
        )

        logger.info(f"Response data: {response.json()}")


if __name__ == "__main__":
    send_request()
