from pathlib import Path

import numpy as np

from models.config_loader import (
    load_aircraft_parameters,
)

from models.controls import ControlInput

from models.aerodynamics import (
    calculate_aerodynamics,
)


def load_params():

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    return load_aircraft_parameters(
        root / "config" / "aircraft.yaml"
    )


def case_1_zero_alpha():

    params = load_params()

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.5,
    )

    result = calculate_aerodynamics(
        u=25.0,
        v=0.0,
        w=0.0,

        p=0.0,
        q=0.0,
        r=0.0,

        altitude=100.0,

        controls=controls,
        params=params,
    )

    print("=" * 60)
    print("CASE 1 — ZERO ANGLE OF ATTACK")
    print("=" * 60)

    print(
        f"V       = {result.airspeed:.6f} m/s"
    )

    print(
        f"alpha   = "
        f"{np.rad2deg(result.alpha):.6f} deg"
    )

    print(
        f"beta    = "
        f"{np.rad2deg(result.beta):.6f} deg"
    )

    print(f"CL      = {result.CL:.6f}")
    print(f"CD      = {result.CD:.6f}")
    print(f"CY      = {result.CY:.6f}")

    print(
        f"\nLift    = {result.lift:.6f} N"
    )

    print(
        f"Drag    = {result.drag:.6f} N"
    )

    print(
        f"Side    = {result.side_force:.6f} N"
    )

    print("\nBody forces [X Y Z]:")
    print(result.forces_body)

    print("\nBody moments [L M N]:")
    print(result.moments_body)

    # --------------------------------------------
    # Expected coefficients
    # --------------------------------------------

    assert np.isclose(
        result.CL,
        params.CL0,
        atol=1e-10,
    )

    assert np.isclose(
        result.CD,
        params.CD0,
        atol=1e-10,
    )

    assert np.isclose(
        result.CY,
        0.0,
        atol=1e-10,
    )

    # --------------------------------------------
    # Expected forces
    # --------------------------------------------

    expected_lift = (
        result.dynamic_pressure
        * params.wing_area
        * params.CL0
    )

    expected_drag = (
        result.dynamic_pressure
        * params.wing_area
        * params.CD0
    )

    assert np.isclose(
        result.lift,
        expected_lift,
        atol=1e-10,
    )

    assert np.isclose(
        result.drag,
        expected_drag,
        atol=1e-10,
    )

    # At alpha = 0:
    # X = -D
    # Z = -L

    assert np.isclose(
        result.forces_body[0],
        -result.drag,
        atol=1e-10,
    )

    assert np.isclose(
        result.forces_body[2],
        -result.lift,
        atol=1e-10,
    )

    print("\nCase 1 PASSED.")


def case_2_positive_alpha():

    params = load_params()

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.5,
    )

    alpha = np.deg2rad(5.0)

    V = 25.0

    u = V * np.cos(alpha)
    w = V * np.sin(alpha)

    result = calculate_aerodynamics(
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

    print("\n" + "=" * 60)
    print("CASE 2 — POSITIVE ANGLE OF ATTACK")
    print("=" * 60)

    print(
        f"V       = {result.airspeed:.6f} m/s"
    )

    print(
        f"alpha   = "
        f"{np.rad2deg(result.alpha):.6f} deg"
    )

    print(
        f"CL      = {result.CL:.6f}"
    )

    print(
        f"CD      = {result.CD:.6f}"
    )

    print(
        f"Lift    = {result.lift:.6f} N"
    )

    print(
        f"Drag    = {result.drag:.6f} N"
    )

    assert np.isclose(
        result.alpha,
        alpha,
        atol=1e-10,
    )

    assert result.CL > params.CL0

    assert result.CD > params.CD0

    print("\nCase 2 PASSED.")


def case_3_elevator_effect():

    params = load_params()

    alpha = np.deg2rad(4.0)

    V = 25.0

    u = V * np.cos(alpha)
    w = V * np.sin(alpha)

    neutral = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.5,
    )

    elevator_down = ControlInput(
        aileron=0.0,
        elevator=np.deg2rad(-5.0),
        rudder=0.0,
        throttle=0.5,
    )

    neutral_result = calculate_aerodynamics(
        u, 0.0, w,
        0.0, 0.0, 0.0,
        100.0,
        neutral,
        params,
    )

    elevator_result = calculate_aerodynamics(
        u, 0.0, w,
        0.0, 0.0, 0.0,
        100.0,
        elevator_down,
        params,
    )

    print("\n" + "=" * 60)
    print("CASE 3 — ELEVATOR EFFECT")
    print("=" * 60)

    print(
        f"Neutral CL       = "
        f"{neutral_result.CL:.6f}"
    )

    print(
        f"Elevator CL      = "
        f"{elevator_result.CL:.6f}"
    )

    print(
        f"Neutral Cm       = "
        f"{neutral_result.Cm:.6f}"
    )

    print(
        f"Elevator Cm      = "
        f"{elevator_result.Cm:.6f}"
    )

    assert (
        elevator_result.CL
        > neutral_result.CL
    )

    assert (
        elevator_result.Cm
        > neutral_result.Cm
    )

    print("\nCase 3 PASSED.")


def case_4_sideslip():

    params = load_params()

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.5,
    )

    result = calculate_aerodynamics(
        u=25.0,
        v=5.0,
        w=0.0,

        p=0.0,
        q=0.0,
        r=0.0,

        altitude=100.0,

        controls=controls,
        params=params,
    )

    print("\n" + "=" * 60)
    print("CASE 4 — SIDESLIP")
    print("=" * 60)

    print(
        f"Beta    = "
        f"{np.rad2deg(result.beta):.6f} deg"
    )

    print(
        f"CY      = "
        f"{result.CY:.6f}"
    )

    print(
        f"Cl      = "
        f"{result.Cl:.6f}"
    )

    print(
        f"Cn      = "
        f"{result.Cn:.6f}"
    )

    assert result.beta > 0.0

    assert result.CY < 0.0

    print("\nCase 4 PASSED.")


def case_5_zero_airspeed():

    params = load_params()

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.5,
    )

    result = calculate_aerodynamics(
        u=0.0,
        v=0.0,
        w=0.0,

        p=0.0,
        q=0.0,
        r=0.0,

        altitude=100.0,

        controls=controls,
        params=params,
    )

    print("\n" + "=" * 60)
    print("CASE 5 — ZERO AIRSPEED")
    print("=" * 60)

    print(
        f"V       = "
        f"{result.airspeed:.6f} m/s"
    )

    print(
        f"Lift    = "
        f"{result.lift:.6f} N"
    )

    print(
        f"Drag    = "
        f"{result.drag:.6f} N"
    )

    print(
        f"Side    = "
        f"{result.side_force:.6f} N"
    )

    assert result.airspeed == 0.0
    assert result.lift == 0.0
    assert result.drag == 0.0
    assert result.side_force == 0.0

    assert np.allclose(
        result.forces_body,
        np.zeros(3),
        atol=1e-10,
    )

    assert np.allclose(
        result.moments_body,
        np.zeros(3),
        atol=1e-10,
    )

    print("\nCase 5 PASSED.")


def main():

    case_1_zero_alpha()
    case_2_positive_alpha()
    case_3_elevator_effect()
    case_4_sideslip()
    case_5_zero_airspeed()

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL AERODYNAMIC TESTS PASSED."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()