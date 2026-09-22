from dataclasses import dataclass

import numpy as np

from models.aircraft import AircraftParameters
from models.aero_state import (
    calculate_aerodynamic_state,
)
from models.controls import ControlInput


@dataclass(frozen=True)
class AerodynamicResult:
    """
    Complete aerodynamic calculation result.

    Forces:
        [X, Y, Z] in body coordinates [N]

    Moments:
        [L, M, N] in body coordinates [N*m]

    Angles:
        alpha, beta [rad]

    Coefficients:
        CL, CD, CY, Cl, Cm, Cn
    """

    forces_body: np.ndarray
    moments_body: np.ndarray

    airspeed: float
    alpha: float
    beta: float
    density: float
    dynamic_pressure: float

    CL: float
    CD: float
    CY: float

    Cl: float
    Cm: float
    Cn: float

    lift: float
    drag: float
    side_force: float


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
    Calculate aerodynamic coefficients, forces and moments.

    Parameters
    ----------
    u, v, w : float
        Body-frame velocity components [m/s].

    p, q, r : float
        Body angular rates [rad/s].

        These are accepted by the interface now so that
        rate-dependent aerodynamic terms can be added later.

    altitude : float
        Altitude above the reference [m].

    controls : ControlInput
        Aileron/elevator/rudder/throttle commands.

    params : AircraftParameters
        Aircraft configuration.

    Returns
    -------
    AerodynamicResult
        Aerodynamic state, coefficients, forces and moments.
    """

    # ---------------------------------------------------------
    # Calculate aerodynamic state
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Limit controls to physically meaningful values
    # ---------------------------------------------------------

    controls = controls.clipped()

    # ---------------------------------------------------------
    # Zero-airspeed handling
    #
    # At V = 0:
    # q_dynamic = 0
    #
    # so aerodynamic forces and moments should be zero.
    # ---------------------------------------------------------

    if V < 1e-6:

        zero_vector = np.zeros(3)

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
        )

    # ---------------------------------------------------------
    # Longitudinal aerodynamic coefficients
    # ---------------------------------------------------------

    CL = (
        params.CL0
        + params.CL_alpha * alpha
        + params.CL_delta_e * controls.elevator
    )

    CD = (
        params.CD0
        + params.CD_alpha * abs(alpha)
        + params.CD_alpha2 * alpha**2
    )

    Cm = (
        params.Cm0
        + params.Cm_alpha * alpha
        + params.Cm_delta_e * controls.elevator
    )

    # ---------------------------------------------------------
    # Lateral-directional coefficients
    # ---------------------------------------------------------

    CY = (
        params.CY_beta * beta
        + params.CY_delta_r * controls.rudder
    )

    Cl = (
        params.Cl_beta * beta
        + params.Cl_delta_a * controls.aileron
    )

    Cn = (
        params.Cn_beta * beta
        + params.Cn_delta_r * controls.rudder
    )

    # ---------------------------------------------------------
    # Aerodynamic force magnitudes
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Convert lift/drag into body X/Z
    #
    # Body axes:
    #   x = forward
    #   z = down
    #
    # Lift is upward and drag opposes the relative wind.
    # ---------------------------------------------------------

    X_aero = (
        -drag * np.cos(alpha)
        + lift * np.sin(alpha)
    )

    Y_aero = side_force

    Z_aero = (
        -drag * np.sin(alpha)
        - lift * np.cos(alpha)
    )

    # ---------------------------------------------------------
    # Aerodynamic moments
    # ---------------------------------------------------------

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

    return AerodynamicResult(
        forces_body=np.array([
            X_aero,
            Y_aero,
            Z_aero,
        ]),

        moments_body=np.array([
            roll_moment,
            pitch_moment,
            yaw_moment,
        ]),

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
    )