from pathlib import Path

import yaml

from models.aircraft import AircraftParameters


def load_aircraft_parameters(
    filepath: str | Path,
) -> AircraftParameters:
    """
    Load aircraft parameters from a YAML file.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Aircraft configuration not found: {filepath}"
        )

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Aircraft configuration must contain a YAML mapping."
        )

    inertia = data["inertia"]
    geometry = data["geometry"]

    aero = data["aerodynamics"]
    longitudinal = aero["longitudinal"]
    lateral = aero["lateral"]

    propulsion = data["propulsion"]

    return AircraftParameters(

        name=data["name"],

        mass=float(data["mass"]),

        gravity=float(
            data.get("gravity", 9.81)
        ),

        Ixx=float(inertia["Ixx"]),
        Iyy=float(inertia["Iyy"]),
        Izz=float(inertia["Izz"]),
        Ixz=float(inertia["Ixz"]),

        wing_area=float(
            geometry["wing_area"]
        ),

        wing_span=float(
            geometry["wing_span"]
        ),

        mean_chord=float(
            geometry["mean_chord"]
        ),

        CL0=float(
            longitudinal["CL0"]
        ),

        CL_alpha=float(
            longitudinal["CL_alpha"]
        ),

        CL_delta_e=float(
            longitudinal["CL_delta_e"]
        ),

        CD0=float(
            longitudinal["CD0"]
        ),

        CD_alpha=float(
            longitudinal["CD_alpha"]
        ),

        CD_alpha2=float(
            longitudinal["CD_alpha2"]
        ),

        Cm0=float(
            longitudinal["Cm0"]
        ),

        Cm_alpha=float(
            longitudinal["Cm_alpha"]
        ),

        Cm_delta_e=float(
            longitudinal["Cm_delta_e"]
        ),

        CY_beta=float(
            lateral["CY_beta"]
        ),

        CY_delta_r=float(
            lateral["CY_delta_r"]
        ),

        Cl_beta=float(
            lateral["Cl_beta"]
        ),

        Cl_delta_a=float(
            lateral["Cl_delta_a"]
        ),

        Cn_beta=float(
            lateral["Cn_beta"]
        ),

        Cn_delta_r=float(
            lateral["Cn_delta_r"]
        ),

        max_thrust=float(
            propulsion["max_thrust"]
        ),
    )