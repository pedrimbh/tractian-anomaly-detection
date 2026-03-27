import random
import time

from locust import HttpUser, between, task
from requests.exceptions import ConnectionError as RequestsConnectionError

SERIES_IDS = [f"sensor_{i}" for i in range(1, 11)]
N_TRAINING_POINTS = 50
MAX_RETRIES = 3


def _post_with_retry(client, url: str, json: dict) -> None:
    """Envia POST retentando até MAX_RETRIES vezes em respostas 5xx ou erros de conexão.

    Backoff exponencial: 100ms, 200ms entre tentativas.
    Erros 4xx não são retentados — são falhas do cliente (dados inválidos).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(url, json=json)
            if response.status_code < 500:
                return
        except RequestsConnectionError:
            pass  # conexão resetada pelo servidor — retentar
        if attempt < MAX_RETRIES - 1:
            time.sleep(0.1 * (2 ** attempt))


class SensorUser(HttpUser):
    wait_time = between(0.05, 0.5)

    def on_start(self) -> None:
        self.series_id = random.choice(SERIES_IDS)
        timestamps = list(range(N_TRAINING_POINTS))
        values = [random.uniform(10.0, 20.0) for _ in range(N_TRAINING_POINTS)]
        _post_with_retry(
            self.client,
            f"/fit/{self.series_id}",
            {"timestamps": timestamps, "values": values},
        )

    @task(1)
    def predict(self) -> None:
        _post_with_retry(
            self.client,
            f"/predict/{self.series_id}",
            {"timestamp": int(time.time()), "value": random.uniform(10.0, 20.0)},
        )

    @task(1)
    def retrain(self) -> None:
        timestamps = list(range(N_TRAINING_POINTS))
        values = [random.uniform(10.0, 20.0) for _ in range(N_TRAINING_POINTS)]
        _post_with_retry(
            self.client,
            f"/fit/{self.series_id}",
            {"timestamps": timestamps, "values": values},
        )
