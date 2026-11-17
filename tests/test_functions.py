import math

import numpy as np
import pytest

from ml_from_scratch.functions import cross_entropy, log_softmax, softmax


def test_softmax_rows_sum_to_one():
    out = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert out.sum(axis=-1) == pytest.approx([1.0, 1.0])


def test_softmax_equal_scores_give_equal_probabilities():
    assert softmax(np.zeros(4)) == pytest.approx([0.25] * 4)


def test_softmax_is_stable_for_huge_numbers():
    out = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(out))
    assert out.sum() == pytest.approx(1.0)


def test_log_softmax_matches_log_of_softmax():
    x = np.array([[0.5, -1.0, 2.0]])
    assert log_softmax(x) == pytest.approx(np.log(softmax(x)))


def test_cross_entropy_known_value():
    # p(correct) = e^2 / (e^2 + 2), loss = -log(p)
    expected = math.log(1 + 2 * math.exp(-2))
    assert cross_entropy(np.array([[2.0, 0.0, 0.0]]), np.array([0])) == pytest.approx(expected)


def test_cross_entropy_for_a_blind_guess_is_log_of_classes():
    logits = np.zeros((5, 10))
    assert cross_entropy(logits, np.arange(5)) == pytest.approx(math.log(10))
