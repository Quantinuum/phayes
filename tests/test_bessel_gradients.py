import jax
from jax import numpy as jnp
import numpy as np
import pytest

import phayes


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
def test_bessel_gradients_at_uniform_distribution(inverse, compiled, vectorized):
    function = phayes.inverse_bessel_ratio if inverse else phayes.bessel_ratio
    differentiate = jax.grad(function)
    if vectorized:
        differentiate = jax.vmap(differentiate)
    if compiled:
        differentiate = jax.jit(differentiate)

    argument = jnp.zeros(3) if vectorized else 0.0
    # I1(k)/I0(k) = k/2 + O(k^3); its inverse is 2r + O(r^3).
    expected = 2.0 if inverse else 0.5
    np.testing.assert_allclose(differentiate(argument), expected, rtol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_bessel_gradients_away_from_origin(compiled):
    forward = jax.grad(phayes.bessel_ratio)
    inverse = jax.grad(phayes.inverse_bessel_ratio)
    if compiled:
        forward = jax.jit(forward)
        inverse = jax.jit(inverse)

    # The asymptotic branch is 1 - 1/(2k).
    np.testing.assert_allclose(forward(1000.0), 0.5e-6, rtol=1e-6)
    # Differentiate the documented large-r rational approximation analytically.
    s = 0.1
    expected_inverse = (2 - 2 * s - 3 * s**2) / (2 * s - s**2 - s**3) ** 2
    np.testing.assert_allclose(inverse(0.9), expected_inverse, rtol=1e-5)
