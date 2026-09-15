from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftParameters:
    """
    Aircraft physical, geometric, aerodynamic,
    and propulsion parameters.

    All internal units are SI.
    """

    # ---------------------------------------------------------
    # Basic aircraft properties
    # ---------------------------------------------------------

    name: str
    mass: float
    gravity: float

    # ---------------------------------------------------------
    # Inertia
    # ---------------------------------------------------------

    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    wing_area: float
    wing_span: float
    mean_chord: float

    # ---------------------------------------------------------
    # Longitudinal aerodynamics
    # ---------------------------------------------------------

    CL0: float
    CL_alpha: float
    CL_delta_e: float

    CD0: float
    CD_alpha: float
    CD_alpha2: float

    Cm0: float
    Cm_alpha: float
    Cm_delta_e: float

    # ---------------------------------------------------------
    # Lateral-directional aerodynamics
    # ---------------------------------------------------------

    CY_beta: float
    CY_delta_r: float

    Cl_beta: float
    Cl_delta_a: float

    Cn_beta: float
    Cn_delta_r: float

    # ---------------------------------------------------------
    # Propulsion
    # ---------------------------------------------------------

    max_thrust: float