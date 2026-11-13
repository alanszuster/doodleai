import numpy as np
import pytest

import classifier
from calibration import fit_temperature, nll, temper


def overconfident_model(n=30000, classes=6, true_temperature=2.0, seed=0):
    """Probabilities that are too sharp: labels follow softmax(z / T), the model says softmax(z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=(n, classes))

    def softmax(a):
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    truth = softmax(z / true_temperature)
    labels = np.array([rng.choice(classes, p=p) for p in truth])
    return softmax(z), labels


def test_the_fit_recovers_the_known_temperature_and_lowers_the_loss():
    probs, labels = overconfident_model()
    temperature = fit_temperature(probs, labels)
    assert temperature == pytest.approx(2.0, abs=0.15)
    assert nll(temper(probs, temperature), labels) < nll(probs, labels)


def test_an_underconfident_model_gets_a_temperature_below_one():
    probs, labels = overconfident_model(true_temperature=0.5)
    assert fit_temperature(probs, labels) < 1.0


def test_temper_keeps_a_distribution_and_the_ranking():
    probs = np.array([[0.7, 0.2, 0.1], [0.05, 0.05, 0.9]])
    for temperature in (0.5, 1.0, 2.5):
        out = temper(probs, temperature)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert (out.argmax(axis=1) == probs.argmax(axis=1)).all()
    assert temper(probs, 1.0) is not None and np.array_equal(temper(probs, 1.0), probs)


def test_the_api_applies_the_same_formula_as_the_training_side():
    probs = np.array([0.6, 0.3, 0.1])
    assert np.allclose(classifier.temper(probs, 1.7), temper(probs[None, :], 1.7)[0])
    assert classifier.temper(probs, 1.0) is probs


@pytest.mark.parametrize(
    "content", [None, "not json", '{"temperature": "hot"}', '{"temperature": 99}']
)
def test_a_missing_or_nonsense_temperature_means_no_change(tmp_path, content):
    path = tmp_path / "model_card.json"
    if content is not None:
        path.write_text(content)
    assert classifier.read_temperature(path) == 1.0


def test_a_valid_temperature_is_read_from_the_card(tmp_path):
    path = tmp_path / "model_card.json"
    path.write_text('{"temperature": 1.8, "run_id": "x"}')
    assert classifier.read_temperature(path) == 1.8
