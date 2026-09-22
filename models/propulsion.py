import numpy as np

from models.aircraft import AircraftParameters


def calculate_thrust(
    throttle: float,
    airspeed: float,
    params: AircraftParameters,
) -> np.ndarray:
    """
    Simple thrust model.

    Parameters
    ----------
    throttle : float
        Normalized throttle command [0, 1].

    airspeed : float
        Aircraft airspeed [m/s].

    params : AircraftParameters
        Aircraft configuration.

    Returns
    -------
    np.ndarray
        Thrust vector in body coordinates [N].

    Notes
    -----
    The current Level 1A model assumes thrust is aligned
    with the body x-axis and has a simple linear throttle
    relationship.

    Airspeed is included in the interface so a more realistic
    propeller model can be added later.
    """

    del airspeed  # Not yet used in this simplified model.

    throttle = float(
        np.clip(
            throttle,
            0.0,
            1.0,
        )
    )

    thrust = (
        throttle
        * params.max_thrust
    )

    return np.array([
        thrust,
        0.0,
        0.0,
    ])