from dataclasses import dataclass

import numpy as np

from models.atmosphere import air_density


@dataclass(frozen=True)
class AerodynamicState:
    """
    Quantities required by the aerodynamic model.

    All angles are in radians.
    """

    airspeed: float
    alpha: float
    beta: float
    density: float
    dynamic_pressure: float


def calculate_aerodynamic_state(
    u: float,
    v: float,
    w: float,
    altitude: float,
) -> AerodynamicState:
    """
    Calculate aerodynamic state from body-frame velocity
    and altitude.

    Parameters
    ----------
    u : float
        Body x velocity [m/s].

    v : float
        Body y velocity [m/s].

    w : float
        Body z velocity [m/s].

    altitude : float
        Altitude above reference [m].

    Returns
    -------
    AerodynamicState
        Airspeed, angle of attack, sideslip,
        density and dynamic pressure.
    """

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    altitude = max(
        0.0,
        float(altitude),
    )

    # ---------------------------------------------------------
    # Airspeed magnitude
    # ---------------------------------------------------------

    airspeed = float(
        np.sqrt(
            u**2
            + v**2
            + w**2
        )
    )

    # ---------------------------------------------------------
    # Protect against zero/near-zero airspeed
    # ---------------------------------------------------------

    if airspeed < 1e-6:
        return AerodynamicState(
            airspeed=0.0,
            alpha=0.0,
            beta=0.0,
            density=air_density(altitude),
            dynamic_pressure=0.0,
        )

    # ---------------------------------------------------------
    # Angle of attack
    # ---------------------------------------------------------

    alpha = float(
        np.arctan2(w, u)
    )

    # ---------------------------------------------------------
    # Sideslip angle
    # ---------------------------------------------------------

    beta_argument = np.clip(
        v / airspeed,
        -1.0,
        1.0,
    )

    beta = float(
        np.arcsin(beta_argument)
    )

    # ---------------------------------------------------------
    # Atmospheric density
    # ---------------------------------------------------------

    rho = float(
        air_density(altitude)
    )

    # ---------------------------------------------------------
    # Dynamic pressure
    # ---------------------------------------------------------

    dynamic_pressure = (
        0.5
        * rho
        * airspeed**2
    )

    return AerodynamicState(
        airspeed=airspeed,
        alpha=alpha,
        beta=beta,
        density=rho,
        dynamic_pressure=dynamic_pressure,
    )