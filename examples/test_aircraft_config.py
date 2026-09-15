from pathlib import Path

from models.config_loader import load_aircraft_parameters


def main():

    # Find the project root directory
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # Path to aircraft configuration
    config_file = (
        project_root
        / "config"
        / "aircraft.yaml"
    )

    # Load the aircraft parameters
    aircraft = load_aircraft_parameters(
        config_file
    )

    print("=" * 50)
    print("AEROGNC AIRCRAFT CONFIGURATION")
    print("=" * 50)

    print(f"Name            : {aircraft.name}")
    print(f"Mass            : {aircraft.mass:.3f} kg")

    print("\nInertia")
    print(f"Ixx             : {aircraft.Ixx:.6f}")
    print(f"Iyy             : {aircraft.Iyy:.6f}")
    print(f"Izz             : {aircraft.Izz:.6f}")
    print(f"Ixz             : {aircraft.Ixz:.6f}")

    print("\nGeometry")
    print(
        f"Wing area       : "
        f"{aircraft.wing_area:.3f} m^2"
    )

    print(
        f"Wing span       : "
        f"{aircraft.wing_span:.3f} m"
    )

    print(
        f"Mean chord      : "
        f"{aircraft.mean_chord:.3f} m"
    )

    print("\nAerodynamics")
    print(f"CL0             : {aircraft.CL0}")
    print(f"CL_alpha        : {aircraft.CL_alpha}")
    print(f"CL_delta_e      : {aircraft.CL_delta_e}")

    print("\nPropulsion")
    print(
        f"Maximum thrust  : "
        f"{aircraft.max_thrust:.3f} N"
    )

    print("\nConfiguration loaded successfully.")


if __name__ == "__main__":
    main()