import jax
from jax import numpy as jnp
import numpy as np
import pytest

import phayes


@pytest.mark.parametrize("mode", ["fourier", "von_mises", "adaptive"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_callable_update_matches_per_shot_rates(mode, compiled, constant):
    measurements = jnp.array([0, 1, 0])
    exponents = jnp.array([1, 2, 4])
    phases = jnp.array([0.2, 0.6, 0.4])

    if constant:
        error_rate = lambda k: 0.25
        rates = np.full(3, 0.25)
    else:
        error_rate = lambda k: 1 - jnp.exp(-k / 10)
        rates = 1 - np.exp(-np.array([1, 2, 4]) / 10)

    if mode == "fourier":
        prior = jnp.zeros((2, 20))

        def update(prior, m, k, beta, error_rate):
            return phayes.fourier_update(prior, m, k, beta, error_rate)

    elif mode == "von_mises":
        prior = (0.3, 0.8)

        def update(prior, m, k, beta, error_rate):
            return phayes.von_mises_update(*prior, m, k, beta, error_rate)

    else:
        # Small capacity exercises a Fourier-to-von-Mises transition too.
        prior = phayes.init(3)

        def update(prior, m, k, beta, error_rate):
            return phayes.update(prior, m, k, beta, error_rate)

    expected = prior
    for m, k, beta, rate in zip(measurements, exponents, phases, rates):
        expected = update(expected, m, k, beta, float(rate))

    def batch_update(prior):
        return update(prior, measurements, exponents, phases, error_rate)

    if compiled:
        batch_update = jax.jit(batch_update)
    actual = batch_update(prior)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("noise", [0.0, 0.25, 1.0])
def test_callable_fourier_update_matches_likelihood(compiled, noise):
    prior = jnp.zeros((2, 16))
    measurements = np.array([0, 1, 0])
    exponents = np.array([1, 2, 3])
    phases = np.array([0.2, 0.6, 0.4])

    def update(prior):
        return phayes.fourier_update(
            prior,
            measurements,
            exponents,
            phases,
            lambda k: noise,
        )

    if compiled:
        update = jax.jit(update)
    posterior = update(prior)

    # The periodic grid integrates this finite trigonometric polynomial exactly.
    grid = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
    likelihood = np.prod(
        (
            1
            + (1 - noise)
            * np.cos(
                exponents[:, None] * grid
                + phases[:, None]
                - measurements[:, None] * np.pi
            )
        )
        / 2,
        axis=0,
    )
    expected_pdf = likelihood / (2 * np.pi * likelihood.mean())
    actual_pdf = phayes.fourier_pdf(jnp.asarray(grid), posterior)
    np.testing.assert_allclose(actual_pdf, expected_pdf, rtol=1e-5, atol=1e-6)
