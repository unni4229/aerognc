import numpy as np

from dynamics.kinematics import (
    body_to_ned_dcm,
    euler_rate_matrix,
)


def inertia_matrix(
    Ixx: float,
    Iyy: float,
    Izz: float,
    Ixz: float,
) -> np.ndarray:

    return np.array([
        [Ixx, 0.0, -Ixz],
        [0.0, Iyy, 0.0],
        [-Ixz, 0.0, Izz],
    ], dtype=float)


def rigid_body_derivatives(
    state: np.ndarray,
    forces_body: np.ndarray,
    moments_body: np.ndarray,
    mass: float,
    gravity: float,
    Ixx: float,
    Iyy: float,
    Izz: float,
    Ixz: float,
) -> np.ndarray:
    """
    Calculate nonlinear 6-DOF rigid-body state derivatives.

    State:
        [pn, pe, pd,
         u, v, w,
         phi, theta, psi,
         p, q, r]

    Forces:
        [X, Y, Z] in body coordinates [N]

    Moments:
        [L, M, N] in body coordinates [N m]

    Returns:
        12-element state derivative.
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    forces_body = np.asarray(
        forces_body,
        dtype=float,
    )

    moments_body = np.asarray(
        moments_body,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    if state.shape != (12,):
        raise ValueError(
            "State must have shape (12,)."
        )

    if forces_body.shape != (3,):
        raise ValueError(
            "Forces must have shape (3,)."
        )

    if moments_body.shape != (3,):
        raise ValueError(
            "Moments must have shape (3,)."
        )

    if mass <= 0.0:
        raise ValueError(
            "Aircraft mass must be positive."
        )

    # ---------------------------------------------------------
    # Unpack state
    # ---------------------------------------------------------

    (
        pn,
        pe,
        pd,

        u,
        v,
        w,

        phi,
        theta,
        psi,

        p,
        q,
        r,
    ) = state

    # ---------------------------------------------------------
    # Body velocity vector
    # ---------------------------------------------------------

    velocity_body = np.array([
        u,
        v,
        w,
    ])

    # ---------------------------------------------------------
    # Body angular velocity vector
    # ---------------------------------------------------------

    omega_body = np.array([
        p,
        q,
        r,
    ])

    # ---------------------------------------------------------
    # Position kinematics
    #
    # V_NED = C_bn * V_body
    # ---------------------------------------------------------

    C_bn = body_to_ned_dcm(
        phi,
        theta,
        psi,
    )

    velocity_ned = (
        C_bn @ velocity_body
    )

    pn_dot = velocity_ned[0]
    pe_dot = velocity_ned[1]
    pd_dot = velocity_ned[2]

    # ---------------------------------------------------------
    # Gravity
    # ---------------------------------------------------------

    gravity_ned = np.array([
        0.0,
        0.0,
        gravity,
    ])

    gravity_body_vector = (
        C_bn.T @ gravity_ned
    )

    # ---------------------------------------------------------
    # Translational dynamics
    #
    # m(V_dot + omega x V) =
    #       F + m*g
    #
    # therefore:
    #
    # V_dot =
    #       F/m
    #       + g
    #       - omega x V
    # ---------------------------------------------------------

    coriolis_body = np.cross(
        omega_body,
        velocity_body,
    )

    velocity_dot_body = (
        forces_body / mass
        + gravity_body_vector
        - coriolis_body
    )

    u_dot = velocity_dot_body[0]
    v_dot = velocity_dot_body[1]
    w_dot = velocity_dot_body[2]

    # ---------------------------------------------------------
    # Euler-angle kinematics
    # ---------------------------------------------------------

    T_euler = euler_rate_matrix(
        phi,
        theta,
    )

    euler_dot = (
        T_euler @ omega_body
    )

    phi_dot = euler_dot[0]
    theta_dot = euler_dot[1]
    psi_dot = euler_dot[2]

    # ---------------------------------------------------------
    # Inertia matrix
    # ---------------------------------------------------------

    I = inertia_matrix(
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=Ixz,
    )

    # ---------------------------------------------------------
    # Rotational dynamics
    #
    # I*omega_dot +
    #     omega x (I*omega) = M
    #
    # therefore:
    #
    # omega_dot =
    #     I^-1 [
    #         M - omega x (I*omega)
    #     ]
    # ---------------------------------------------------------

    angular_momentum = (
        I @ omega_body
    )

    rotational_coupling = np.cross(
        omega_body,
        angular_momentum,
    )

    omega_dot = np.linalg.solve(
        I,
        moments_body
        - rotational_coupling,
    )

    p_dot = omega_dot[0]
    q_dot = omega_dot[1]
    r_dot = omega_dot[2]

    # ---------------------------------------------------------
    # Assemble complete state derivative
    # ---------------------------------------------------------

    return np.array([
        pn_dot,
        pe_dot,
        pd_dot,

        u_dot,
        v_dot,
        w_dot,

        phi_dot,
        theta_dot,
        psi_dot,

        p_dot,
        q_dot,
        r_dot,
    ])