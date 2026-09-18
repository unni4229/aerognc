import numpy as np


# Standard sea-level density [kg/m^3]
SEA_LEVEL_DENSITY = 1.225

# Simple atmospheric scale height [m]
SCALE_HEIGHT = 8500.0


def air_density(altitude_m: float) -> float:
    """
    Calculate approximate atmospheric density.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level [m].

    Returns
    -------
    float
        Air density [kg/m^3].

    Notes
    -----
    This is a simple exponential atmosphere model used
    for the initial AeroGNC simulation.
    """

    altitude_m = max(
        0.0,
        float(altitude_m),
    )

    rho = (
        SEA_LEVEL_DENSITY
        * np.exp(
            -altitude_m / SCALE_HEIGHT
        )
    )

    return float(rho)