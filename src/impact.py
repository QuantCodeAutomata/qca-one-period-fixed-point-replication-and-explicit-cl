# Custom — Context7 found no library equivalent (paper methodology)
import numpy as np
from typing import Callable


def linear_impact(lam: float) -> Callable:
    """I(x) = lambda*x"""
    return lambda x: lam * np.asarray(x, dtype=float)


def sqrt_impact(lam: float) -> Callable:
    """I(x) = lambda*sign(x)*sqrt(|x|)"""
    def I(x):
        x = np.asarray(x, dtype=float)
        return lam * np.sign(x) * np.sqrt(np.abs(x))
    return I


def linear_exec_cost(varphi: float) -> Callable:
    """phi(x) = varphi*x  (linear execution cost)"""
    return lambda x: varphi * np.asarray(x, dtype=float)


def fig2_exec_cost(lam: float, varphi: float) -> Callable:
    """Figure-2 piecewise execution cost:
    phi(x) = varphi*sign(x)*sqrt(|x|) for |x| <= 4*varphi^2/lambda^2
           = (lambda/2)*x otherwise
    """
    threshold = 4 * varphi**2 / lam**2
    def phi(x):
        x = np.asarray(x, dtype=float)
        mask = np.abs(x) <= threshold
        result = np.where(mask,
                          varphi * np.sign(x) * np.sqrt(np.abs(x)),
                          (lam / 2.0) * x)
        return result
    return phi
