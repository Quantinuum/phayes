import jax
from jax import numpy as jnp
import numpy as np
import pytest

import phayes


@pytest.mark.parametrize("mode", ["fourier", "adaptive"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("exponent", [1, 2, 3, 10])
@pytest.mark.parametrize("measurement", [0, 1])
def test_evidence_respects_fourier_bandwidth(mode, compiled, exponent, measurement):
    coefficients = jnp.array([[0.06, 0.03], [0.02, -0.01]])
    phase = 0.4

    def error_rate(k):
        return 1 - jnp.exp(-k / 12)

    if mode == "fourier":
        prior = coefficients
        evidence = phayes.fourier_evidence
    else:
        prior = phayes.PhayesState(True, coefficients, (0.0, 0.0))
        evidence = phayes.evidence

    def evaluate(prior):
        return evidence(prior, measurement, exponent, phase, error_rate)

    if compiled:
        evaluate = jax.jit(evaluate)

    # Integrate the finite Fourier prior and likelihood on a periodic grid.
    grid = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
    prior_pdf = (
        1 / (2 * np.pi)
        + 0.06 * np.cos(grid)
        + 0.03 * np.cos(2 * grid)
        + 0.02 * np.sin(grid)
        - 0.01 * np.sin(2 * grid)
    )
    likelihood = (
        1
        + np.exp(-exponent / 12) * np.cos(exponent * grid + phase - measurement * np.pi)
    ) / 2
    expected = 2 * np.pi * np.mean(prior_pdf * likelihood)
    np.testing.assert_allclose(evaluate(prior), expected, rtol=1e-6, atol=1e-7)
