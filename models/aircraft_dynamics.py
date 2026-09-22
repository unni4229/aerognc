import numpy as np

from models.aircraft import AircraftParameters
from models.controls import ControlInput
from models.aerodynamics import (
    calculate_aerodynamics,
)
from models.propulsion import (
    calculate_thrust,
)
from dynamics.rigid_body import (
    rigid_body_derivatives,
)


def aircraft_derivatives(
    state: np.ndarray,
    controls: ControlInput,
    params: AircraftParameters,
) -> np.ndarray:
    """
    Complete nonlinear aircraft plant.

    Inputs
    ------
    state:
        12-state aircraft vector

        [pn, pe, pd,
         u, v, w,
         phi, theta, psi,
         p, q, r]

    controls:
        [aileron, elevator, rudder, throttle]

    params:
        Aircraft configuration.

    Returns
    -------
    xdot:
        12-state derivative.
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    if state.shape != (12,):
        raise ValueError(
            "Aircraft state must have shape (12,)."
        )

    controls = controls.clipped()

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
    # Altitude
    #
    # NED convention:
    # Down is positive
    #
    # Therefore:
    # altitude = -pd
    # ---------------------------------------------------------

    altitude = max(
        0.0,
        -pd,
    )

    # ---------------------------------------------------------
    # Aerodynamics
    # ---------------------------------------------------------

    aero = calculate_aerodynamics(
        u=u,
        v=v,
        w=w,

        p=p,
        q=q,
        r=r,

        altitude=altitude,

        controls=controls,
        params=params,
    )

    # ---------------------------------------------------------
    # Propulsion
    # ---------------------------------------------------------

    thrust = calculate_thrust(
        throttle=controls.throttle,
        airspeed=aero.airspeed,
        params=params,
    )

    # ---------------------------------------------------------
    # Total body forces
    # ---------------------------------------------------------

    total_forces_body = (
        aero.forces_body
        + thrust
    )

    # ---------------------------------------------------------
    # Total body moments
    #
    # At this stage propulsion does not create moments.
    # ---------------------------------------------------------

    total_moments_body = (
        aero.moments_body
    )

    # ---------------------------------------------------------
    # Rigid-body equations
    # ---------------------------------------------------------

    return rigid_body_derivatives(
        state=state,

        forces_body=total_forces_body,

        moments_body=total_moments_body,

        mass=params.mass,

        gravity=params.gravity,

        Ixx=params.Ixx,
        Iyy=params.Iyy,
        Izz=params.Izz,
        Ixz=params.Ixz,
    )