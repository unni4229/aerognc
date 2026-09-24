"""
AeroGNC aircraft configuration loader.
"""

from pathlib import Path

import yaml

from models.aircraft import AircraftParameters


def load_aircraft_parameters(
    filepath: str | Path,
) -> AircraftParameters:
    """
    Load aircraft parameters from aircraft.yaml.
    """

    filepath = Path(filepath)

    # ---------------------------------------------------------
    # Check configuration file
    # ---------------------------------------------------------

    if not filepath.exists():
        raise FileNotFoundError(
            f"Aircraft configuration not found: {filepath}"
        )

    # ---------------------------------------------------------
    # Read YAML
    # ---------------------------------------------------------

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Aircraft configuration must be a YAML mapping."
        )

    # ---------------------------------------------------------
    # Extract sections
    # ---------------------------------------------------------

    inertia = data["inertia"]

    geometry = data["geometry"]

    aerodynamics = data["aerodynamics"]

    longitudinal = aerodynamics["longitudinal"]

    lateral = aerodynamics["lateral"]

    propulsion = data["propulsion"]

    # ---------------------------------------------------------
    # Create AircraftParameters
    # ---------------------------------------------------------

    return AircraftParameters(

        # =====================================================
        # Basic properties
        # =====================================================

        name=str(
            data["name"]
        ),

        mass=float(
            data["mass"]
        ),

        gravity=float(
            data.get(
                "gravity",
                9.81,
            )
        ),

        # =====================================================
        # Inertia
        # =====================================================

        Ixx=float(
            inertia["Ixx"]
        ),

        Iyy=float(
            inertia["Iyy"]
        ),

        Izz=float(
            inertia["Izz"]
        ),

        Ixz=float(
            inertia["Ixz"]
        ),

        # =====================================================
        # Geometry
        # =====================================================

        wing_area=float(
            geometry["wing_area"]
        ),

        wing_span=float(
            geometry["wing_span"]
        ),

        mean_chord=float(
            geometry["mean_chord"]
        ),

        # =====================================================
        # Lift
        # =====================================================

        CL0=float(
            longitudinal["CL0"]
        ),

        CL_alpha=float(
            longitudinal["CL_alpha"]
        ),

        CL_delta_e=float(
            longitudinal["CL_delta_e"]
        ),

        # =====================================================
        # Drag
        # =====================================================

        CD0=float(
            longitudinal["CD0"]
        ),

        CD_alpha=float(
            longitudinal["CD_alpha"]
        ),

        CD_alpha2=float(
            longitudinal["CD_alpha2"]
        ),

        # =====================================================
        # Pitching moment
        # =====================================================

        Cm0=float(
            longitudinal["Cm0"]
        ),

        Cm_alpha=float(
            longitudinal["Cm_alpha"]
        ),

        Cm_delta_e=float(
            longitudinal["Cm_delta_e"]
        ),

        # =====================================================
        # NEW: pitch-rate damping
        # =====================================================

        Cm_q=float(
            longitudinal["Cm_q"]
        ),

        # =====================================================
        # Side force
        # =====================================================

        CY_beta=float(
            lateral["CY_beta"]
        ),

        CY_delta_r=float(
            lateral["CY_delta_r"]
        ),

        # =====================================================
        # Rolling moment
        # =====================================================

        Cl_beta=float(
            lateral["Cl_beta"]
        ),

        Cl_delta_a=float(
            lateral["Cl_delta_a"]
        ),

        # =====================================================
        # NEW: roll-rate damping
        # =====================================================

        Cl_p=float(
            lateral["Cl_p"]
        ),

        # =====================================================
        # Yawing moment
        # =====================================================

        Cn_beta=float(
            lateral["Cn_beta"]
        ),

        Cn_delta_r=float(
            lateral["Cn_delta_r"]
        ),

        # =====================================================
        # NEW: yaw-rate damping
        # =====================================================

        Cn_r=float(
            lateral["Cn_r"]
        ),

        # =====================================================
        # Propulsion
        # =====================================================

        max_thrust=float(
            propulsion["max_thrust"]
        ),
    )