from pathlib import Path

import numpy as np

from models.config_loader import (
    load_aircraft_parameters,
)

from models.controls import (
    ControlInput,
)

from models.aircraft_dynamics import (
    aircraft_derivatives,
)


def main():

    # ---------------------------------------------------------
    # Load parameters
    # ---------------------------------------------------------

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    params = load_aircraft_parameters(
        root / "config" / "aircraft.yaml"
    )

    # ---------------------------------------------------------
    # Initial aircraft state
    #
    # Position:
    #   0,0,-100
    #
    # Velocity:
    #   25 m/s forward
    #
    # Attitude:
    #   level, heading north
    #
    # Rates:
    #   zero
    # ---------------------------------------------------------

    state = np.array([
        0.0,
        0.0,
        -100.0,

        25.0,
        0.0,
        0.0,

        0.0,
        0.0,
        0.0,

        0.0,
        0.0,
        0.0,
    ])

    # ---------------------------------------------------------
    # Control input
    # ---------------------------------------------------------

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.50,
    )

    # ---------------------------------------------------------
    # Calculate complete state derivative
    # ---------------------------------------------------------

    xdot = aircraft_derivatives(
        state=state,
        controls=controls,
        params=params,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    labels = [
        "pn_dot",
        "pe_dot",
        "pd_dot",

        "u_dot",
        "v_dot",
        "w_dot",

        "phi_dot",
        "theta_dot",
        "psi_dot",

        "p_dot",
        "q_dot",
        "r_dot",
    ]

    print("=" * 60)
    print("AEROGNC COMPLETE AIRCRAFT PLANT")
    print("=" * 60)

    for label, value in zip(
        labels,
        xdot,
    ):

        print(
            f"{label:10s}: "
            f"{value: .6f}"
        )

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    assert xdot.shape == (12,)

    assert np.all(
        np.isfinite(xdot)
    )

    # Position rate should initially be northward.
    assert np.isclose(
        xdot[0],
        25.0,
        atol=1e-10,
    )

    assert np.isclose(
        xdot[1],
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        xdot[2],
        0.0,
        atol=1e-10,
    )

    print(
        "\nComplete aircraft plant test PASSED."
    )
    zero_throttle = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.0,
    )

    full_throttle = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=1.0,
    )

if __name__ == "__main__":
    main()
    