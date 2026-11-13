"""Temperature scaling: make the model's confidences match how often it is right.

The network ends in a softmax. Training with label smoothing keeps it from ever being fully sure,
so its percentages are systematically too low (or, after other changes, too high). One number,
the temperature T, fixes that without touching the weights or the ranking: with probabilities
``p``, the calibrated ones are ``p ** (1 / T)`` renormalised, which equals ``softmax(logits / T)``.
T is fitted on the validation split (never the test split), stored in the model card, and applied
by the API. numpy only.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def temper(probs, temperature: float):
    """``probs`` rescaled by ``temperature``; T = 1 returns them unchanged."""
    probs = np.asarray(probs, dtype=np.float64)
    if temperature == 1.0:
        return probs
    scaled = np.power(np.clip(probs, EPS, 1.0), 1.0 / temperature)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def nll(probs, labels) -> float:
    """Mean negative log-likelihood of the true class."""
    probs = np.asarray(probs, dtype=np.float64)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def fit_temperature(probs, labels, *, low: float = 0.25, high: float = 4.0) -> float:
    """The T that minimises the NLL of ``temper(probs, T)`` (a convex problem in 1 / T)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)

    def loss(beta):  # beta = 1 / T
        return nll(temper(probs, 1.0 / beta), labels)

    lo, hi = 1.0 / high, 1.0 / low
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - golden * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + golden * (hi - lo)
            fb = loss(b)
    return float(1.0 / ((lo + hi) / 2.0))
