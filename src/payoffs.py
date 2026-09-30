# Custom — Context7 found no library equivalent (paper methodology)
import numpy as np
from typing import Callable


def call_payoff(x: np.ndarray, K: float, N: float = 1.0) -> np.ndarray:
    """Call option payoff V(x) = N*(x-K)^+"""
    return N * np.maximum(x - K, 0.0)


def put_payoff(x: np.ndarray, K: float, N: float = 1.0) -> np.ndarray:
    """Put option payoff V(x) = N*(K-x)^+"""
    return N * np.maximum(K - x, 0.0)


def digital_call_payoff(x: np.ndarray, K: float, N: float = 1.0) -> np.ndarray:
    """Digital call: N * 1_{x > K}"""
    return N * (x > K).astype(float)


def make_call(K: float, N: float = 1.0) -> Callable:
    return lambda x: call_payoff(np.asarray(x, dtype=float), K, N)


def make_put(K: float, N: float = 1.0) -> Callable:
    return lambda x: put_payoff(np.asarray(x, dtype=float), K, N)


def make_digital_call(K: float, N: float = 1.0) -> Callable:
    return lambda x: digital_call_payoff(np.asarray(x, dtype=float), K, N)
