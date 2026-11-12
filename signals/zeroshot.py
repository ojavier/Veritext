"""Zero-shot language-model signals.

The module deliberately does not depend on a particular model library.  A
caller only needs to provide the log-probability assigned to each observed
token by a model.  Log probabilities are expected to use natural logarithms.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias

LogProbabilities: TypeAlias = Iterable[float]


def _exp(value: float) -> float:
    """Exponentiate while preserving infinity for representable inputs."""

    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _finite_values(values: LogProbabilities, *, name: str) -> list[float]:
    """Materialize and validate a sequence of token log probabilities."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers")

    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numbers") from error

    if not result:
        raise ValueError(f"{name} must contain at least one token")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def perplexity(log_probabilities: LogProbabilities) -> float:
    """Return token-level perplexity from natural-log probabilities.

    For token log probabilities ``l_i``, perplexity is
    ``exp(-mean(l_i))``.  Supplying log probabilities rather than
    probabilities avoids underflow for long segments.
    """

    values = _finite_values(log_probabilities, name="log_probabilities")
    return _exp(-math.fsum(values) / len(values))


def cross_perplexity(
    target_probabilities: Iterable[float],
    predicted_probabilities: Iterable[float],
) -> float:
    """Return cross-perplexity between two discrete distributions.

    ``target_probabilities`` is the reference distribution ``p`` and
    ``predicted_probabilities`` is the model distribution ``q``.  The value
    is ``exp(H(p, q))`` where ``H(p, q) = -sum(p * log(q))``.  Both
    distributions must have equal, non-zero length and are normalized
    internally, which also permits unnormalized non-negative weights.
    """

    target = _finite_values(target_probabilities, name="target_probabilities")
    predicted = _finite_values(
        predicted_probabilities, name="predicted_probabilities"
    )
    if len(target) != len(predicted):
        raise ValueError("probability distributions must have equal length")
    if any(value < 0 for value in target + predicted):
        raise ValueError("probabilities must be non-negative")

    target_total = math.fsum(target)
    predicted_total = math.fsum(predicted)
    if target_total <= 0 or predicted_total <= 0:
        raise ValueError("probability distributions must have a positive sum")

    cross_entropy = 0.0
    for target_value, predicted_value in zip(target, predicted):
        if target_value == 0:
            continue
        if predicted_value == 0:
            return math.inf
        cross_entropy -= (
            target_value / target_total
        ) * math.log(predicted_value / predicted_total)
    return _exp(cross_entropy)


def token_cross_perplexity(
    reference_log_probabilities: LogProbabilities,
    model_log_probabilities: LogProbabilities,
) -> float:
    """Return cross-perplexity for the same tokens scored by two models.

    This is useful when comparing a text's likelihood under a reference model
    and a candidate model.  The reference model supplies token weights
    ``p_i = exp(reference_log_probabilities[i])`` and the candidate supplies
    ``q_i``; the calculation is delegated to :func:`cross_perplexity`.
    """

    reference = _finite_values(
        reference_log_probabilities, name="reference_log_probabilities"
    )
    model = _finite_values(model_log_probabilities, name="model_log_probabilities")
    if len(reference) != len(model):
        raise ValueError("log-probability sequences must have equal length")

    reference_probabilities = [math.exp(value) for value in reference]
    model_probabilities = [math.exp(value) for value in model]
    return cross_perplexity(reference_probabilities, model_probabilities)


__all__ = [
    "LogProbabilities",
    "cross_perplexity",
    "perplexity",
    "token_cross_perplexity",
]