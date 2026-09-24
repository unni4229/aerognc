from pathlib import Path

import numpy as np

from models.config_loader import (
    load_aircraft_parameters,
)

from models.controls import ControlInput

from models.aerodynamics import (
    calculate_aerodynamics,
)


def main():

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    params = load_aircraft_parameters(
        root / "config" / "aircraft.yaml"
    )

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.0,
    )

    # ---------------------------------------------------------
    # Use a normal flight condition
    # ---------------------------------------------------------

    alpha = np.deg2rad(4.0553)

    V = 25.0

    u = V * np.cos(alpha)
    w = V * np.sin(alpha)

    # ---------------------------------------------------------
    # Zero pitch rate
    # ---------------------------------------------------------

    no_rate = calculate_aerodynamics(
        u=u,
        v=0.0,
        w=w,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude=100.0,
        controls=controls,
        params=params,
    )

    # ---------------------------------------------------------
    # Positive pitch rate
    # ---------------------------------------------------------

    positive_q = calculate_aerodynamics(
        u=u,
        v=0.0,
        w=w,
        p=0.0,
        q=np.deg2rad(10.0),
        r=0.0,
        altitude=100.0,
        controls=controls,
        params=params,
    )

    # ---------------------------------------------------------
    # Positive roll rate
    # ---------------------------------------------------------

    positive_p = calculate_aerodynamics(
        u=u,
        v=0.0,
        w=w,
        p=np.deg2rad(10.0),
        q=0.0,
        r=0.0,
        altitude=100.0,
        controls=controls,
        params=params,
    )

    # ---------------------------------------------------------
    # Positive yaw rate
    # ---------------------------------------------------------

    positive_r = calculate_aerodynamics(
        u=u,
        v=0.0,
        w=w,
        p=0.0,
        q=0.0,
        r=np.deg2rad(10.0),
        altitude=100.0,
        controls=controls,
        params=params,
    )

    print("=" * 60)
    print("AERODYNAMIC DAMPING TEST")
    print("=" * 60)

    print("\nPitch damping")

    print(
        f"Cm at q = 0      : "
        f"{no_rate.Cm:.6f}"
    )

    print(
        f"Cm at q > 0      : "
        f"{positive_q.Cm:.6f}"
    )

    print("\nRoll damping")

    print(
        f"Cl at p = 0      : "
        f"{no_rate.Cl:.6f}"
    )

    print(
        f"Cl at p > 0      : "
        f"{positive_p.Cl:.6f}"
    )

    print("\nYaw damping")

    print(
        f"Cn at r = 0      : "
        f"{no_rate.Cn:.6f}"
    )

    print(
        f"Cn at r > 0      : "
        f"{positive_r.Cn:.6f}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert (
        positive_q.Cm
        < no_rate.Cm
    )

    assert (
        positive_p.Cl
        < no_rate.Cl
    )

    assert (
        positive_r.Cn
        < no_rate.Cn
    )

    print(
        "\nDamping sign tests PASSED."
    )


if __name__ == "__main__":
    main()