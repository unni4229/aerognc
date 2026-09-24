"""
AeroGNC aerodynamic model.

This module converts aircraft velocity, angular rates, control inputs,
and atmospheric conditions into aerodynamic coefficients, forces,
and moments.

Coordinate convention
---------------------
Navigation frame:
    N = North
    E = East
    D = Down

Body frame:
    x = forward
    y = right
    z = down

All internal angles are radians.
All forces are Newtons.
All moments are N*m.
"""

from dataclasses import dataclass

import numpy as np

from models.aircraft import AircraftParameters
from models.aero_state import calculate_aerodynamic_state
from models.controls import ControlInput


@dataclass(frozen=True)
class AerodynamicResult:
    """
    Complete aerodynamic calculation result.
    """

    # ---------------------------------------------------------
    # Body-frame forces
    # ---------------------------------------------------------
    # [X, Y, Z] [N]
    # ---------------------------------------------------------

    forces_body: np.ndarray

    # ---------------------------------------------------------
    # Body-frame aerodynamic moments
    # ---------------------------------------------------------
    # [L, M, N] [N*m]
    # ---------------------------------------------------------

    moments_body: np.ndarray

    # ---------------------------------------------------------
    # Aerodynamic state
    # ---------------------------------------------------------

    airspeed: float
    alpha: float
    beta: float
    density: float
    dynamic_pressure: float

    # ---------------------------------------------------------
    # Aerodynamic coefficients
    # ---------------------------------------------------------

    CL: float
    CD: float
    CY: float

    Cl: float
    Cm: float
    Cn: float

    # ---------------------------------------------------------
    # Force magnitudes
    # ---------------------------------------------------------

    lift: float
    drag: float
    side_force: float

    # ---------------------------------------------------------
    # Optional nondimensional angular rates
    # ---------------------------------------------------------

    p_hat: float
    q_hat: float
    r_hat: float


def calculate_aerodynamics(
    u: float,
    v: float,
    w: float,
    p: float,
    q: float,
    r: float,
    altitude: float,
    controls: ControlInput,
    params: AircraftParameters,
) -> AerodynamicResult:
    """
    Calculate aerodynamic coefficients, forces, and moments.

    Parameters
    ----------
    u : float
        Body-frame forward velocity [m/s].

    v : float
        Body-frame lateral velocity [m/s].

    w : float
        Body-frame vertical/down velocity [m/s].

    p : float
        Body roll rate [rad/s].

    q : float
        Body pitch rate [rad/s].

    r : float
        Body yaw rate [rad/s].

    altitude : float
        Altitude above the reference [m].

    controls : ControlInput
        Aircraft control inputs.

        aileron  [rad]
        elevator [rad]
        rudder   [rad]
        throttle [-]

    params : AircraftParameters
        Aircraft physical and aerodynamic parameters.

    Returns
    -------
    AerodynamicResult
        Aerodynamic state, coefficients, forces,
        and moments.

    Notes
    -----
    This is an initial aerodynamic model intended for
    nonlinear simulation and control-development purposes.

    Static aerodynamic terms:
        CL(alpha, elevator)
        CD(alpha)
        CY(beta, rudder)
        Cl(beta, aileron)
        Cm(alpha, elevator)
        Cn(beta, rudder)

    Dynamic damping terms:
        Cl_p * p_hat
        Cm_q * q_hat
        Cn_r * r_hat

    where:

        p_hat = p*b/(2V)
        q_hat = q*c/(2V)
        r_hat = r*b/(2V)

    The current model assumes zero wind.
    """

    # =========================================================
    # 1. Validate aircraft parameters
    # =========================================================

    if params.mass <= 0.0:
        raise ValueError(
            "Aircraft mass must be positive."
        )

    if params.wing_area <= 0.0:
        raise ValueError(
            "Wing area must be positive."
        )

    if params.wing_span <= 0.0:
        raise ValueError(
            "Wing span must be positive."
        )

    if params.mean_chord <= 0.0:
        raise ValueError(
            "Mean aerodynamic chord must be positive."
        )

    # =========================================================
    # 2. Validate state inputs
    # =========================================================

    values = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "q": q,
        "r": r,
        "altitude": altitude,
    }

    for name, value in values.items():

        if not np.isfinite(value):

            raise ValueError(
                f"{name} contains a non-finite value."
            )

    # =========================================================
    # 3. Limit control inputs
    # =========================================================

    controls = controls.clipped()

    # =========================================================
    # 4. Calculate aerodynamic state
    # =========================================================

    aero_state = calculate_aerodynamic_state(
        u=u,
        v=v,
        w=w,
        altitude=altitude,
    )

    V = aero_state.airspeed
    alpha = aero_state.alpha
    beta = aero_state.beta
    rho = aero_state.density
    q_dynamic = aero_state.dynamic_pressure

    # =========================================================
    # 5. Zero-airspeed condition
    # =========================================================
    #
    # At V = 0:
    #
    #   q_dynamic = 0
    #
    # Therefore aerodynamic forces and moments are zero.
    #
    # This also avoids division by zero in p_hat, q_hat, r_hat.
    # =========================================================

    if V < 1e-6:

        zero_vector = np.zeros(
            3,
            dtype=float,
        )

        return AerodynamicResult(
            forces_body=zero_vector.copy(),

            moments_body=zero_vector.copy(),

            airspeed=0.0,
            alpha=0.0,
            beta=0.0,

            density=rho,
            dynamic_pressure=0.0,

            CL=0.0,
            CD=0.0,
            CY=0.0,

            Cl=0.0,
            Cm=0.0,
            Cn=0.0,

            lift=0.0,
            drag=0.0,
            side_force=0.0,

            p_hat=0.0,
            q_hat=0.0,
            r_hat=0.0,
        )

    # =========================================================
    # 6. Nondimensional angular rates
    # =========================================================
    #
    # p_hat = p*b/(2V)
    # q_hat = q*c/(2V)
    # r_hat = r*b/(2V)
    #
    # These are dimensionless.
    # =========================================================

    p_hat = (
        p
        * params.wing_span
        / (2.0 * V)
    )

    q_hat = (
        q
        * params.mean_chord
        / (2.0 * V)
    )

    r_hat = (
        r
        * params.wing_span
        / (2.0 * V)
    )

    # =========================================================
    # 7. Longitudinal aerodynamic coefficients
    # =========================================================

    # ---------------------------------------------------------
    # Lift coefficient
    #
    # CL = CL0
    #      + CL_alpha * alpha
    #      + CL_delta_e * elevator
    # ---------------------------------------------------------

    CL = (
        params.CL0
        + params.CL_alpha * alpha
        + params.CL_delta_e
        * controls.elevator
    )

    # ---------------------------------------------------------
    # Drag coefficient
    #
    # CD = CD0
    #      + CD_alpha * |alpha|
    #      + CD_alpha2 * alpha^2
    # ---------------------------------------------------------

    CD = (
        params.CD0
        + params.CD_alpha * abs(alpha)
        + params.CD_alpha2 * alpha**2
    )

    # ---------------------------------------------------------
    # Pitching-moment coefficient
    #
    # Cm = Cm0
    #      + Cm_alpha * alpha
    #      + Cm_q * q_hat
    #      + Cm_delta_e * elevator
    # ---------------------------------------------------------

    Cm = (
        params.Cm0
        + params.Cm_alpha * alpha
        + params.Cm_q * q_hat
        + params.Cm_delta_e
        * controls.elevator
    )

    # =========================================================
    # 8. Lateral-directional coefficients
    # =========================================================

    # ---------------------------------------------------------
    # Side-force coefficient
    #
    # CY = CY_beta * beta
    #      + CY_delta_r * rudder
    # ---------------------------------------------------------

    CY = (
        params.CY_beta * beta
        + params.CY_delta_r
        * controls.rudder
    )

    # ---------------------------------------------------------
    # Rolling-moment coefficient
    #
    # Cl = Cl_beta * beta
    #      + Cl_p * p_hat
    #      + Cl_delta_a * aileron
    # ---------------------------------------------------------

    Cl = (
        params.Cl_beta * beta
        + params.Cl_p * p_hat
        + params.Cl_delta_a
        * controls.aileron
    )

    # ---------------------------------------------------------
    # Yawing-moment coefficient
    #
    # Cn = Cn_beta * beta
    #      + Cn_r * r_hat
    #      + Cn_delta_r * rudder
    # ---------------------------------------------------------

    Cn = (
        params.Cn_beta * beta
        + params.Cn_r * r_hat
        + params.Cn_delta_r
        * controls.rudder
    )

    # =========================================================
    # 9. Aerodynamic force magnitudes
    # =========================================================

    lift = (
        q_dynamic
        * params.wing_area
        * CL
    )

    drag = (
        q_dynamic
        * params.wing_area
        * CD
    )

    side_force = (
        q_dynamic
        * params.wing_area
        * CY
    )

    # =========================================================
    # 10. Convert aerodynamic forces to body axes
    # =========================================================
    #
    # Aerodynamic lift/drag are naturally defined relative
    # to the relative wind.
    #
    # Body convention:
    #       x = forward
    #       y = right
    #       z = down
    #
    # Therefore positive lift gives a negative Z-body force.
    # =========================================================

    X_aero = (
        -drag * np.cos(alpha)
        + lift * np.sin(alpha)
    )

    Y_aero = side_force

    Z_aero = (
        -drag * np.sin(alpha)
        - lift * np.cos(alpha)
    )

    # =========================================================
    # 11. Aerodynamic moments
    # =========================================================

    roll_moment = (
        q_dynamic
        * params.wing_area
        * params.wing_span
        * Cl
    )

    pitch_moment = (
        q_dynamic
        * params.wing_area
        * params.mean_chord
        * Cm
    )

    yaw_moment = (
        q_dynamic
        * params.wing_area
        * params.wing_span
        * Cn
    )

    # =========================================================
    # 12. Construct result
    # =========================================================

    return AerodynamicResult(
        forces_body=np.array(
            [
                X_aero,
                Y_aero,
                Z_aero,
            ],
            dtype=float,
        ),

        moments_body=np.array(
            [
                roll_moment,
                pitch_moment,
                yaw_moment,
            ],
            dtype=float,
        ),

        airspeed=V,
        alpha=alpha,
        beta=beta,
        density=rho,
        dynamic_pressure=q_dynamic,

        CL=CL,
        CD=CD,
        CY=CY,

        Cl=Cl,
        Cm=Cm,
        Cn=Cn,

        lift=lift,
        drag=drag,
        side_force=side_force,

        p_hat=p_hat,
        q_hat=q_hat,
        r_hat=r_hat,
    )