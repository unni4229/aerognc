import numpy as np

from models.aero_state import (
    calculate_aerodynamic_state,
)


def test_level_flight():

    aero = calculate_aerodynamic_state(
        u=25.0,
        v=0.0,
        w=0.0,
        altitude=100.0,
    )

    print("=" * 60)
    print("CASE 1 — ZERO ANGLE OF ATTACK / ZERO SIDESLIP")
    print("=" * 60)

    print(
        f"Airspeed          : "
        f"{aero.airspeed:.6f} m/s"
    )

    print(
        f"Alpha             : "
        f"{np.rad2deg(aero.alpha):.6f} deg"
    )

    print(
        f"Beta              : "
        f"{np.rad2deg(aero.beta):.6f} deg"
    )

    print(
        f"Density           : "
        f"{aero.density:.6f} kg/m^3"
    )

    print(
        f"Dynamic pressure  : "
        f"{aero.dynamic_pressure:.6f} Pa"
    )

    # Expected
    assert np.isclose(
        aero.airspeed,
        25.0,
        atol=1e-10,
    )

    assert np.isclose(
        aero.alpha,
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        aero.beta,
        0.0,
        atol=1e-10,
    )

    expected_q = (
        0.5
        * aero.density
        * 25.0**2
    )

    assert np.isclose(
        aero.dynamic_pressure,
        expected_q,
        atol=1e-10,
    )

    print("\nCase 1 PASSED.")


def test_positive_angle_of_attack():

    aero = calculate_aerodynamic_state(
        u=25.0,
        v=0.0,
        w=25.0,
        altitude=100.0,
    )

    print("\n" + "=" * 60)
    print("CASE 2 — POSITIVE ANGLE OF ATTACK")
    print("=" * 60)

    print(
        f"Airspeed          : "
        f"{aero.airspeed:.6f} m/s"
    )

    print(
        f"Alpha             : "
        f"{np.rad2deg(aero.alpha):.6f} deg"
    )

    print(
        f"Beta              : "
        f"{np.rad2deg(aero.beta):.6f} deg"
    )

    assert np.isclose(
        aero.airspeed,
        np.sqrt(25.0**2 + 25.0**2),
        atol=1e-10,
    )

    assert np.isclose(
        np.rad2deg(aero.alpha),
        45.0,
        atol=1e-10,
    )

    assert np.isclose(
        aero.beta,
        0.0,
        atol=1e-10,
    )

    print("\nCase 2 PASSED.")


def test_positive_sideslip():

    aero = calculate_aerodynamic_state(
        u=25.0,
        v=5.0,
        w=0.0,
        altitude=100.0,
    )

    print("\n" + "=" * 60)
    print("CASE 3 — POSITIVE SIDESLIP")
    print("=" * 60)

    print(
        f"Airspeed          : "
        f"{aero.airspeed:.6f} m/s"
    )

    print(
        f"Alpha             : "
        f"{np.rad2deg(aero.alpha):.6f} deg"
    )

    print(
        f"Beta              : "
        f"{np.rad2deg(aero.beta):.6f} deg"
    )

    assert np.isclose(
        aero.alpha,
        0.0,
        atol=1e-10,
    )

    assert aero.beta > 0.0

    expected_beta = np.arcsin(
        5.0 / np.sqrt(25.0**2 + 5.0**2)
    )

    assert np.isclose(
        aero.beta,
        expected_beta,
        atol=1e-10,
    )

    print("\nCase 3 PASSED.")


def test_zero_airspeed():

    aero = calculate_aerodynamic_state(
        u=0.0,
        v=0.0,
        w=0.0,
        altitude=100.0,
    )

    print("\n" + "=" * 60)
    print("CASE 4 — ZERO AIRSPEED")
    print("=" * 60)

    print(
        f"Airspeed          : "
        f"{aero.airspeed:.6f} m/s"
    )

    print(
        f"Alpha             : "
        f"{np.rad2deg(aero.alpha):.6f} deg"
    )

    print(
        f"Beta              : "
        f"{np.rad2deg(aero.beta):.6f} deg"
    )

    print(
        f"Dynamic pressure  : "
        f"{aero.dynamic_pressure:.6f} Pa"
    )

    assert aero.airspeed == 0.0
    assert aero.alpha == 0.0
    assert aero.beta == 0.0
    assert aero.dynamic_pressure == 0.0

    print("\nCase 4 PASSED.")


if __name__ == "__main__":

    test_level_flight()
    test_positive_angle_of_attack()
    test_positive_sideslip()
    test_zero_airspeed()

    print("\nAll aerodynamic-state tests PASSED.")