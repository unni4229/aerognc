import numpy as np

from models.state import AircraftState

from dynamics.kinematics import (
    body_to_ned_dcm,
    body_to_ned,
    ned_to_body,
    euler_rate_matrix,
)


def test_state():

    print("=" * 60)
    print("AEROGNC STATE REPRESENTATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create test aircraft state
    # ---------------------------------------------------------

    state = AircraftState(
        pn=0.0,
        pe=0.0,
        pd=-100.0,

        u=25.0,
        v=0.0,
        w=0.0,

        phi=np.deg2rad(0.0),
        theta=np.deg2rad(0.0),
        psi=np.deg2rad(90.0),

        p=0.0,
        q=0.0,
        r=0.0,
    )

    print("\nAircraft state:")
    print(state)

    # ---------------------------------------------------------
    # Basic state properties
    # ---------------------------------------------------------

    print(
        f"\nAltitude : "
        f"{state.altitude:.2f} m"
    )

    print(
        f"Airspeed : "
        f"{state.airspeed:.2f} m/s"
    )

    # ---------------------------------------------------------
    # State -> vector
    # ---------------------------------------------------------

    vector = state.as_vector()

    print("\nState vector:")
    print(vector)

    # ---------------------------------------------------------
    # Vector -> state
    # ---------------------------------------------------------

    recovered_state = (
        AircraftState.from_vector(vector)
    )

    print("\nRecovered state:")
    print(recovered_state)

    # ---------------------------------------------------------
    # Body -> NED DCM
    # ---------------------------------------------------------

    C_bn = body_to_ned_dcm(
        state.phi,
        state.theta,
        state.psi,
    )

    print("\nBody-to-NED DCM:")
    print(C_bn)

    # ---------------------------------------------------------
    # Body velocity
    # ---------------------------------------------------------

    body_velocity = np.array([
        state.u,
        state.v,
        state.w,
    ])

    print("\nBody velocity:")
    print(body_velocity)

    # ---------------------------------------------------------
    # Body -> NED velocity
    # ---------------------------------------------------------

    ned_velocity = body_to_ned(
        body_velocity,
        state.phi,
        state.theta,
        state.psi,
    )

    print("\nNED velocity:")
    print(ned_velocity)

    # ---------------------------------------------------------
    # NED -> Body velocity
    # ---------------------------------------------------------

    recovered_body_velocity = ned_to_body(
        ned_velocity,
        state.phi,
        state.theta,
        state.psi,
    )

    print("\nRecovered body velocity:")
    print(recovered_body_velocity)

    # ---------------------------------------------------------
    # Euler-rate transformation
    # ---------------------------------------------------------

    T = euler_rate_matrix(
        state.phi,
        state.theta,
    )

    print("\nEuler-rate matrix:")
    print(T)

    body_rates = np.array([
        state.p,
        state.q,
        state.r,
    ])

    euler_rates = T @ body_rates

    print("\nBody angular rates:")
    print(body_rates)

    print("\nEuler angle rates:")
    print(euler_rates)

    # =========================================================
    # VALIDATION TESTS
    # =========================================================

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Test 1: DCM orthogonality
    #
    # For a proper rotation matrix:
    #
    # C^T C = I
    # ---------------------------------------------------------

    identity_check = (
        C_bn.T @ C_bn
    )

    print("\nC^T C:")
    print(identity_check)

    assert np.allclose(
        identity_check,
        np.eye(3),
        atol=1e-10,
    )

    print(
        "\nDCM orthogonality test PASSED."
    )

    # ---------------------------------------------------------
    # Test 2: DCM determinant
    #
    # A proper rotation matrix must have:
    #
    # det(C) = +1
    # ---------------------------------------------------------

    determinant = np.linalg.det(C_bn)

    print(
        f"\nDeterminant of C: "
        f"{determinant:.12f}"
    )

    assert np.isclose(
        determinant,
        1.0,
        atol=1e-10,
    )

    print(
        "DCM determinant test PASSED."
    )

    # ---------------------------------------------------------
    # Test 3: Body -> NED -> Body round trip
    # ---------------------------------------------------------

    assert np.allclose(
        body_velocity,
        recovered_body_velocity,
        atol=1e-10,
    )

    print(
        "Body <-> NED transformation test PASSED."
    )

    # ---------------------------------------------------------
    # Test 4: State vector round trip
    # ---------------------------------------------------------

    recovered_vector = (
        recovered_state.as_vector()
    )

    assert np.allclose(
        vector,
        recovered_vector,
        atol=1e-10,
    )

    print(
        "State vector round-trip test PASSED."
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print(
        "\nState and kinematics tests complete."
    )


if __name__ == "__main__":
    test_state()