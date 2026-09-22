import numpy as np

from simulation.integrator import rk4_step


def test_constant_derivative():

    # dx/dt = 2
    #
    # Exact solution:
    #
    # x(t) = x0 + 2t

    state = np.array([
        1.0
    ])

    dt = 0.1

    def derivative(x):
        return np.array([
            2.0
        ])

    new_state = rk4_step(
        derivative,
        state,
        dt,
    )

    expected = np.array([
        1.2
    ])

    assert np.allclose(
        new_state,
        expected,
        atol=1e-12,
    )

    print(
        "Constant derivative test PASSED."
    )


def test_exponential_growth():

    # dx/dt = x
    #
    # Exact solution:
    #
    # x(t) = x0 * exp(t)

    state = np.array([
        1.0
    ])

    dt = 0.1

    def derivative(x):
        return x

    new_state = rk4_step(
        derivative,
        state,
        dt,
    )

    expected = np.exp(0.1)

    error = abs(
        new_state[0] - expected
    )

    print(
        f"\nRK4 result : {new_state[0]:.12f}"
    )

    print(
        f"Exact      : {expected:.12f}"
    )

    print(
        f"Error      : {error:.12e}"
    )

    assert np.isclose(
        new_state[0],
        expected,
        atol=1e-6,
    )

    print(
        "Exponential-growth test PASSED."
    )


def main():

    print("=" * 60)
    print("RK4 INTEGRATOR TEST")
    print("=" * 60)

    test_constant_derivative()
    test_exponential_growth()

    print(
        "\nALL RK4 TESTS PASSED."
    )


if __name__ == "__main__":
    main()