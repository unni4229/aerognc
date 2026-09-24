from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from models.config_loader import (
    load_aircraft_parameters,
)

from trim.trim_solver import (
    StraightLevelTrimSolver,
)

from simulation.simulator import (
    AircraftSimulator,
)


def main():

    # =========================================================
    # 1. Project paths
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

    # =========================================================
    # 2. Load aircraft parameters
    # =========================================================

    params = load_aircraft_parameters(
        config_file
    )

    # =========================================================
    # 3. Calculate trim
    # =========================================================

    trim_solver = StraightLevelTrimSolver(
        params
    )

    trim = trim_solver.solve(
        airspeed=25.0,
        altitude=100.0,
    )

    if not trim.converged:
        raise RuntimeError(
            "Trim solver did not converge."
        )

    # =========================================================
    # 4. Create simulator
    # =========================================================

    sim = AircraftSimulator(
        params=params,
        dt=0.01,
    )

    # =========================================================
    # 5. Initialize directly at trim
    # =========================================================

    sim.reset(
        trim.state
    )

    # =========================================================
    # 6. Simulate using trim controls
    # =========================================================

    simulation_time = 30.0

    sim.run(
        simulation_time=simulation_time,
        controls=trim.controls,
    )

    # =========================================================
    # 7. Extract results
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
    # 8. Safety checks
    # =========================================================

    if not np.all(
        np.isfinite(states)
    ):
        raise RuntimeError(
            "Non-finite values detected "
            "in trim-validation simulation."
        )

    # =========================================================
    # 9. Calculate derived quantities
    # =========================================================

    north = states[:, 0]
    east = states[:, 1]
    down = states[:, 2]

    altitude = -down

    u = states[:, 3]
    v = states[:, 4]
    w = states[:, 5]

    airspeed = np.sqrt(
        u**2
        + v**2
        + w**2
    )

    roll = np.rad2deg(
        states[:, 6]
    )

    pitch = np.rad2deg(
        states[:, 7]
    )

    yaw = np.rad2deg(
        states[:, 8]
    )

    p_rate = np.rad2deg(
        states[:, 9]
    )

    q_rate = np.rad2deg(
        states[:, 10]
    )

    r_rate = np.rad2deg(
        states[:, 11]
    )

    # =========================================================
    # 10. Tracking errors
    # =========================================================

    airspeed_error = (
        airspeed - trim.airspeed
    )

    altitude_error = (
        altitude - trim.altitude
    )

    pitch_target_deg = np.rad2deg(
        trim.theta
    )

    pitch_error = (
        pitch - pitch_target_deg
    )

    # =========================================================
    # 11. Performance metrics
    # =========================================================

    max_speed_error = np.max(
        np.abs(airspeed_error)
    )

    max_altitude_error = np.max(
        np.abs(altitude_error)
    )

    max_pitch_error = np.max(
        np.abs(pitch_error)
    )

    max_q_rate = np.max(
        np.abs(q_rate)
    )

    final_speed_error = (
        airspeed[-1]
        - trim.airspeed
    )

    final_altitude_error = (
        altitude[-1]
        - trim.altitude
    )

    final_pitch_error = (
        pitch[-1]
        - pitch_target_deg
    )

    # =========================================================
    # 12. Print trim information
    # =========================================================

    print("=" * 70)
    print(
        "AEROGNC NONLINEAR TRIM VALIDATION"
    )
    print("=" * 70)

    print("\nTrim condition")

    print(
        f"Target airspeed : "
        f"{trim.airspeed:.6f} m/s"
    )

    print(
        f"Target altitude : "
        f"{trim.altitude:.6f} m"
    )

    print(
        f"Trim alpha      : "
        f"{np.rad2deg(trim.alpha):.6f} deg"
    )

    print(
        f"Trim theta      : "
        f"{np.rad2deg(trim.theta):.6f} deg"
    )

    print(
        f"Trim elevator   : "
        f"{np.rad2deg(trim.elevator):.6f} deg"
    )

    print(
        f"Trim throttle   : "
        f"{trim.throttle:.6f}"
    )

    # =========================================================
    # 13. Print validation metrics
    # =========================================================

    print("\nValidation metrics")

    print(
        f"Maximum speed error     : "
        f"{max_speed_error:.9f} m/s"
    )

    print(
        f"Maximum altitude error  : "
        f"{max_altitude_error:.9f} m"
    )

    print(
        f"Maximum pitch error     : "
        f"{max_pitch_error:.9f} deg"
    )

    print(
        f"Maximum pitch rate      : "
        f"{max_q_rate:.9f} deg/s"
    )

    print(
        f"\nFinal speed error       : "
        f"{final_speed_error:.9f} m/s"
    )

    print(
        f"Final altitude error    : "
        f"{final_altitude_error:.9f} m"
    )

    print(
        f"Final pitch error       : "
        f"{final_pitch_error:.9f} deg"
    )

    # =========================================================
    # 14. Position results
    # =========================================================

    print("\nFinal position")

    print(
        f"North : {north[-1]:.3f} m"
    )

    print(
        f"East  : {east[-1]:.3f} m"
    )

    print(
        f"Down  : {down[-1]:.3f} m"
    )

    print(
        f"Altitude : {altitude[-1]:.3f} m"
    )

    # =========================================================
    # 15. Trim validation assertions
    # =========================================================

    assert max_speed_error < 1e-6

    assert max_altitude_error < 1e-6

    assert max_pitch_error < 1e-6

    assert max_q_rate < 1e-6

    print(
        "\nTRIM VALIDATION PASSED."
    )

    # =========================================================
    # 16. Plot airspeed
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        airspeed,
        label="Simulated",
    )

    plt.axhline(
        trim.airspeed,
        linestyle="--",
        label="Trim",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Airspeed [m/s]"
    )

    plt.title(
        "Trim Validation — Airspeed"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 17. Plot altitude
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        altitude,
        label="Simulated",
    )

    plt.axhline(
        trim.altitude,
        linestyle="--",
        label="Trim",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Altitude [m]"
    )

    plt.title(
        "Trim Validation — Altitude"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 18. Plot attitude
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        roll,
        label="Roll",
    )

    plt.plot(
        time,
        pitch,
        label="Pitch",
    )

    plt.plot(
        time,
        yaw,
        label="Yaw",
    )

    plt.axhline(
        pitch_target_deg,
        linestyle="--",
        label="Trim Pitch",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Angle [deg]"
    )

    plt.title(
        "Trim Validation — Attitude"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 19. Plot angular rates
    # =========================================================

    plt.figure()

    plt.plot(
        time,
        p_rate,
        label="p",
    )

    plt.plot(
        time,
        q_rate,
        label="q",
    )

    plt.plot(
        time,
        r_rate,
        label="r",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Rate [deg/s]"
    )

    plt.title(
        "Trim Validation — Angular Rates"
    )

    plt.grid(True)
    plt.legend()

    # =========================================================
    # 20. Plot horizontal trajectory
    # =========================================================

    plt.figure()

    plt.plot(
        east,
        north,
    )

    plt.scatter(
        east[0],
        north[0],
        label="Start",
    )

    plt.scatter(
        east[-1],
        north[-1],
        label="End",
    )

    plt.xlabel(
        "East [m]"
    )

    plt.ylabel(
        "North [m]"
    )

    plt.title(
        "Trim Validation — Horizontal Trajectory"
    )

    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()