"""
Aircraft parameter definitions for AeroGNC.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftParameters:
    """
    Physical, geometric, aerodynamic, and propulsion
    parameters for the aircraft.

    All internal units are SI.

    Angles:
        radians

    Mass:
        kg

    Inertia:
        kg*m^2

    Geometry:
        m

    Aerodynamic coefficients:
        dimensionless

    Thrust:
        N
    """

    # =========================================================
    # Basic aircraft properties
    # =========================================================

    name: str
    mass: float
    gravity: float

    # =========================================================
    # Inertia properties
    # =========================================================

    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float

    # =========================================================
    # Aircraft geometry
    # =========================================================

    wing_area: float
    wing_span: float
    mean_chord: float

    # =========================================================
    # Longitudinal aerodynamics
    # =========================================================

    CL0: float
    CL_alpha: float
    CL_delta_e: float

    CD0: float
    CD_alpha: float
    CD_alpha2: float

    Cm0: float
    Cm_alpha: float
    Cm_delta_e: float

    # Pitch-rate damping derivative
    Cm_q: float

    # =========================================================
    # Lateral-directional aerodynamics
    # =========================================================

    CY_beta: float
    CY_delta_r: float

    Cl_beta: float
    Cl_delta_a: float

    # Roll-rate damping derivative
    Cl_p: float

    Cn_beta: float
    Cn_delta_r: float

    # Yaw-rate damping derivative
    Cn_r: float

    # =========================================================
    # Propulsion
    # =========================================================

    max_thrust: float