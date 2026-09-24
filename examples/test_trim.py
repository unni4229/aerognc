from pathlib import Path

import numpy as np

from models.config_loader import (
    load_aircraft_parameters,
)

from trim.trim_solver import (
    StraightLevelTrimSolver,
)


def main():

    # ---------------------------------------------------------
    # Load aircraft parameters
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Create trim solver
    # ---------------------------------------------------------

    solver = StraightLevelTrimSolver(
        params
    )

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    solution = solver.solve(
        airspeed=25.0,
        altitude=100.0,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("=" * 70)
    print("AEROGNC STRAIGHT-AND-LEVEL TRIM")
    print("=" * 70)

    print(
        f"\nTarget airspeed : "
        f"{solution.airspeed:.6f} m/s"
    )

    print(
        f"Target altitude : "
        f"{solution.altitude:.6f} m"
    )

    print("\nTrim state")

    print(
        f"Alpha           : "
        f"{np.rad2deg(solution.alpha):.6f} deg"
    )

    print(
        f"Theta           : "
        f"{np.rad2deg(solution.theta):.6f} deg"
    )

    print("\nTrim controls")

    print(
        f"Elevator        : "
        f"{np.rad2deg(solution.elevator):.6f} deg"
    )

    print(
        f"Throttle        : "
        f"{solution.throttle:.6f}"
    )

    print(
        f"Throttle        : "
        f"{solution.throttle * 100.0:.3f} %"
    )

    print("\nTrim residuals")

    print(
        f"u_dot           : "
        f"{solution.residual[0]: .12e}"
    )

    print(
        f"w_dot           : "
        f"{solution.residual[1]: .12e}"
    )

    print(
        f"q_dot           : "
        f"{solution.residual[2]: .12e}"
    )

    print(
        f"\nSolver cost      : "
        f"{solution.cost:.12e}"
    )

    print(
        f"Solver optimality: "
        f"{solution.optimality:.12e}"
    )

    print(
        f"Converged        : "
        f"{solution.converged}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert solution.converged

    assert np.linalg.norm(
        solution.residual
    ) < 1e-8

    assert (
        0.0
        <= solution.throttle
        <= 1.0
    )

    assert (
        np.deg2rad(-25.0)
        <= solution.elevator
        <= np.deg2rad(25.0)
    )

    print(
        "\nTRIM SOLUTION VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()