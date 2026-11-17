"""Basic building blocks, written from scratch.

Every function here has a plain-English explanation in the comments.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Turn raw scores into probabilities that add up to 1."""
    x = np.asarray(x, dtype=float)
    # Subtracting the biggest number changes nothing in the result,
    # but it stops exp() from overflowing on big inputs.
    shifted = x - x.max(axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(x)), computed in a numerically safe way."""
    x = np.asarray(x, dtype=float)
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Average loss when the right answer is targets[i] for each row i.

    logits:  shape (batch, classes), raw scores from a model.
    targets: shape (batch,), the index of the correct class for each row.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    log_probs = log_softmax(logits, axis=-1)
    picked = log_probs[np.arange(len(targets)), targets]
    return float(-picked.mean())
