import numpy as np
from typing import Callable


def rk4_step(
    derivative_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Perform one fourth-order Runge-Kutta integration step.

    Parameters
    ----------
    derivative_function : callable
        Function that accepts the current state vector and
        returns the corresponding state derivative.

    state : np.ndarray
        Current state vector.

    dt : float
        Integration timestep [s].

    Returns
    -------
    np.ndarray
        State after one RK4 integration step.
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    if dt <= 0.0:
        raise ValueError(
            "Integration timestep must be positive."
        )

    # First evaluation
    k1 = derivative_function(
        state
    )

    # Second evaluation
    k2 = derivative_function(
        state + 0.5 * dt * k1
    )

    # Third evaluation
    k3 = derivative_function(
        state + 0.5 * dt * k2
    )

    # Fourth evaluation
    k4 = derivative_function(
        state + dt * k3
    )

    # RK4 weighted average
    new_state = state + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )

    return new_state