from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from models.config_loader import (
    load_aircraft_parameters,
)

from models.controls import (
    ControlInput,
)

from simulation.simulator import (
    AircraftSimulator,
)


def main():

    # =========================================================
    # 1. Load aircraft parameters
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_file = (
        project_root
        / "config"
        / "aircraft.yaml"
    )

    params = load_aircraft_parameters(
        config_file
    )

    # =========================================================
    # 2. Create simulator
    # =========================================================

    sim = AircraftSimulator(
        params=params,
        dt=0.01,
    )

    # =========================================================
    # 3. Define initial state
    #
    # Position:
    #   N = 0 m
    #   E = 0 m
    #   D = -100 m
    #
    # Velocity:
    #   u = 25 m/s
    #
    # Attitude:
    #   level
    #
    # Angular rates:
    #   zero
    # =========================================================

    initial_state = np.array([
        # Position
        0.0,
        0.0,
        -100.0,

        # Body velocity
        25.0,
        0.0,
        0.0,

        # Euler attitude
        0.0,
        0.0,
        0.0,

        # Body angular rates
        0.0,
        0.0,
        0.0,
    ])

    sim.reset(
        initial_state
    )

    # =========================================================
    # 4. Open-loop control
    # =========================================================

    controls = ControlInput(
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.50,
    )

    # =========================================================
    # 5. Run simulation
    # =========================================================

    simulation_time = 10.0

    sim.run(
        simulation_time=simulation_time,
        controls=controls,
    )

    # =========================================================
    # 6. Extract results
    # =========================================================

    time = np.array([
        sample.time
        for sample in sim.history
    ])

    states = np.array([
        sample.state
        for sample in sim.history
    ])

    # =========================================================
    # 7. Safety check
    # =========================================================

    if not np.all(
        np.isfinite(states)
    ):
        raise RuntimeError(
            "Non-finite values detected "
            "in simulation history."
        )

    # =========================================================
    # 8. Print final state
    # =========================================================

    final_state = states[-1]

    print("=" * 60)
    print("AEROGNC OPEN-LOOP FLIGHT SIMULATION")
    print("=" * 60)

    print(
        f"Simulation time : "
        f"{time[-1]:.3f} s"
    )

    print(
        f"North position  : "
        f"{final_state[0]:.3f} m"
    )

    print(
        f"East position   : "
        f"{final_state[1]:.3f} m"
    )

    print(
        f"Down position   : "
        f"{final_state[2]:.3f} m"
    )

    print(
        f"Altitude        : "
        f"{-final_state[2]:.3f} m"
    )

    print(
        f"u               : "
        f"{final_state[3]:.3f} m/s"
    )

    print(
        f"v               : "
        f"{final_state[4]:.3f} m/s"
    )

    print(
        f"w               : "
        f"{final_state[5]:.3f} m/s"
    )

    print(
        f"Roll            : "
        f"{np.rad2deg(final_state[6]):.3f} deg"
    )

    print(
        f"Pitch           : "
        f"{np.rad2deg(final_state[7]):.3f} deg"
    )

    print(
        f"Yaw             : "
        f"{np.rad2deg(final_state[8]):.3f} deg"
    )

    # =========================================================
    # 9. Plot North-East trajectory
    # =========================================================

    plt.figure()

    plt.plot(
        states[:, 1],
        states[:, 0],
    )

    plt.xlabel(
        "East Position [m]"
    )

    plt.ylabel(
        "North Position [m]"
    )

    plt.title(
        "Open-Loop Horizontal Trajectory"
    )

    plt.grid(True)
    plt.axis("equal")

    # =========================================================
    # 10. Plot altitude
    # =========================================================

    altitude = -states[:, 2]

    plt.figure()

    plt.plot(
        time,
        altitude,
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Altitude [m]"
    )

    plt.title(
        "Aircraft Altitude"
    )

    plt.grid(True)

    # =========================================================
    # 11. Plot body velocities
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        states[:, 3],
        label="u",
    )

    plt.plot(
        time,
        states[:, 4],
        label="v",
    )

    plt.plot(
        time,
        states[:, 5],
        label="w",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Velocity [m/s]"
    )

    plt.title(
        "Body Velocity"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 12. Plot attitude
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        np.rad2deg(states[:, 6]),
        label="Roll",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 7]),
        label="Pitch",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 8]),
        label="Yaw",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Angle [deg]"
    )

    plt.title(
        "Aircraft Attitude"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 13. Plot angular rates
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        np.rad2deg(states[:, 9]),
        label="p",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 10]),
        label="q",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 11]),
        label="r",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Rate [deg/s]"
    )

    plt.title(
        "Aircraft Angular Rates"
    )

    plt.grid(True)
    plt.legend()

    fig = plt.figure()

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        states[:, 1],       # East
        states[:, 0],       # North
        -states[:, 2],      # Altitude
    )

    ax.set_xlabel(
        "East [m]"
    )

    ax.set_ylabel(
        "North [m]"
    )

    ax.set_zlabel(
        "Altitude [m]"
    )

    ax.set_title(
        "3D Aircraft Trajectory"
    )

    plt.show()


if __name__ == "__main__":
    main()