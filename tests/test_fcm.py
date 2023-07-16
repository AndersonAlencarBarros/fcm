import numpy as np
import time
import random
import pytest
from src.fcm import FCM


@pytest.fixture
def observacoes():
    X = np.array(
        [
            [1, 3],
            [2, 5],
            [4, 8],
            [7, 9],
        ]
    )

    return X


@pytest.fixture
def inicializacao():
    u = np.array([[0.8, 0.7, 0.2, 0.1], [0.2, 0.3, 0.8, 0.9]])

    return u


def test_FCM():
    X = np.array(
        [
            [1, 3],
            [2, 5],
            [4, 8],
            [7, 9],
        ]
    )

    u = np.array([[0.8, 0.7, 0.2, 0.1], [0.2, 0.3, 0.8, 0.9]])

    fcm = FCM(n_clusters=2, mu=2)

    start = time.perf_counter()
    fcm.fit(data=X, u=u)
    end = time.perf_counter()

    print()
    print(f"Elapsed = {end - start}s")
    print()

    print("FCM")
    print("centers")
    print(fcm.centers)
    print("u")
    print(fcm.u)
