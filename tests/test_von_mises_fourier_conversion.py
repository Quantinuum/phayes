import jax
import numpy as np
import pytest
from scipy.special import ive

import phayes


@pytest.mark.parametrize("mu", [0.0, 0.7])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 25.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_von_mises_fourier_coefficients(mu, kappa, compiled):
    convert = phayes.von_mises_to_fourier
    if compiled:
        convert = jax.jit(convert, static_argnames="J")
    actual = convert(mu, kappa, J=5)

    orders = np.arange(1, 6)
    amplitudes = ive(orders, kappa) / (np.pi * ive(0, kappa))
    expected = np.stack(
        (amplitudes * np.cos(orders * mu), amplitudes * np.sin(orders * mu))
    )
    assert actual.shape == (2, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
