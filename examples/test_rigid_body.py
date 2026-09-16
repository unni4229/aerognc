import numpy as np

from dynamics.rigid_body import (
    rigid_body_derivatives,
)


def print_state_derivative(
    title,
    xdot,
):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

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

    for label, value in zip(
        labels,
        xdot,
    ):
        print(
            f"{label:10s}: {value: .6f}"
        )


def main():

    # ---------------------------------------------------------
    # Aircraft parameters
    # ---------------------------------------------------------

    mass = 13.5

    gravity = 9.81

    Ixx = 0.8244
    Iyy = 1.135
    Izz = 1.759
    Ixz = 0.1204

    # ---------------------------------------------------------
    # Test state
    #
    # Aircraft at:
    #   zero position
    #   25 m/s forward velocity
    #   zero attitude
    #   zero angular velocity
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
    # CASE 1
    #
    # Gravity only
    # ---------------------------------------------------------

    forces = np.array([
        0.0,
        0.0,
        0.0,
    ])

    moments = np.array([
        0.0,
        0.0,
        0.0,
    ])

    xdot = rigid_body_derivatives(
        state=state,
        forces_body=forces,
        moments_body=moments,
        mass=mass,
        gravity=gravity,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=Ixz,
    )

    print_state_derivative(
        "CASE 1 — GRAVITY ONLY",
        xdot,
    )

    # ---------------------------------------------------------
    # CASE 2
    #
    # Positive X force
    # ---------------------------------------------------------

    forces = np.array([
        10.0,
        0.0,
        0.0,
    ])

    xdot_xforce = rigid_body_derivatives(
        state=state,
        forces_body=forces,
        moments_body=moments,
        mass=mass,
        gravity=gravity,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=Ixz,
    )

    print_state_derivative(
        "CASE 2 — POSITIVE X FORCE",
        xdot_xforce,
    )

    # ---------------------------------------------------------
    # CASE 3
    #
    # Positive pitching moment
    # ---------------------------------------------------------

    forces = np.array([
        0.0,
        0.0,
        0.0,
    ])

    moments = np.array([
        0.0,
        5.0,
        0.0,
    ])

    xdot_pitch = rigid_body_derivatives(
        state=state,
        forces_body=forces,
        moments_body=moments,
        mass=mass,
        gravity=gravity,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=Ixz,
    )

    print_state_derivative(
        "CASE 3 — POSITIVE PITCH MOMENT",
        xdot_pitch,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    expected_u_dot = 10.0 / mass

    assert np.isclose(
        xdot_xforce[3],
        expected_u_dot,
        atol=1e-10,
    )

    expected_q_dot = 5.0 / Iyy

    assert np.isclose(
        xdot_pitch[10],
        expected_q_dot,
        atol=1e-10,
    )

    assert np.isclose(
        xdot[5],
        gravity,
        atol=1e-10,
    )

    print("\nAll rigid-body sanity checks PASSED.")


if __name__ == "__main__":
    main()