from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from models.aircraft import AircraftParameters
from models.controls import ControlInput
from models.aircraft_dynamics import (
    aircraft_derivatives,
)


@dataclass(frozen=True)
class TrimSolution:
    """
    Result of a straight-and-level aircraft trim calculation.
    """

    airspeed: float
    altitude: float

    alpha: float
    theta: float

    elevator: float
    throttle: float

    residual: np.ndarray

    converged: bool
    cost: float
    optimality: float

    state: np.ndarray
    controls: ControlInput


class StraightLevelTrimSolver:
    """
    Solve for straight-and-level aircraft trim.

    Target condition:
        airspeed  = V_target
        altitude  = h_target
        flight path angle = 0 deg

    Unknowns:
        alpha
        elevator
        throttle

    Conditions:
        u_dot = 0
        w_dot = 0
        q_dot = 0
    """

    def __init__(
        self,
        params: AircraftParameters,
    ):

        self.params = params

    def build_state(
        self,
        airspeed: float,
        altitude: float,
        alpha: float,
    ) -> np.ndarray:
        """
        Construct aircraft state for straight-and-level flight.

        For gamma = 0:
            theta = alpha

        We assume:
            phi = 0
            beta = 0
            p = q = r = 0
        """

        if airspeed <= 0.0:
            raise ValueError(
                "Target airspeed must be positive."
            )

        if altitude < 0.0:
            raise ValueError(
                "Altitude cannot be negative "
                "for this trim model."
            )

        theta = alpha

        # Convert airspeed and alpha into body velocities.
        u = (
            airspeed
            * np.cos(alpha)
        )

        w = (
            airspeed
            * np.sin(alpha)
        )

        state = np.array([
            # Position
            0.0,               # pn
            0.0,               # pe
            -altitude,         # pd

            # Body velocity
            u,
            0.0,               # v
            w,

            # Attitude
            0.0,               # phi
            theta,
            0.0,               # psi

            # Angular rates
            0.0,               # p
            0.0,               # q
            0.0,               # r
        ])

        return state

    def build_controls(
        self,
        elevator: float,
        throttle: float,
    ) -> ControlInput:
        """
        Construct trim control input.

        Aileron and rudder are zero because we are solving
        straight, wings-level flight.
        """

        return ControlInput(
            aileron=0.0,
            elevator=elevator,
            rudder=0.0,
            throttle=throttle,
        ).clipped()

    def residual(
        self,
        variables: np.ndarray,
        airspeed: float,
        altitude: float,
    ) -> np.ndarray:
        """
        Calculate normalized trim residuals.

        variables:
            [alpha, elevator, throttle]
        """

        alpha = variables[0]
        elevator = variables[1]
        throttle = variables[2]

        state = self.build_state(
            airspeed=airspeed,
            altitude=altitude,
            alpha=alpha,
        )

        controls = self.build_controls(
            elevator=elevator,
            throttle=throttle,
        )

        xdot = aircraft_derivatives(
            state=state,
            controls=controls,
            params=self.params,
        )

        # -----------------------------------------------------
        # Trim equations
        #
        # u_dot = 0
        # w_dot = 0
        # q_dot = 0
        # -----------------------------------------------------

        return np.array([
            xdot[3],
            xdot[5],
            xdot[10],
        ])

    def solve(
        self,
        airspeed: float = 25.0,
        altitude: float = 100.0,
        alpha_guess_deg: float = 4.0,
        elevator_guess_deg: float = -2.0,
        throttle_guess: float = 0.25,
    ) -> TrimSolution:
        """
        Solve straight-and-level trim.
        """

        # -----------------------------------------------------
        # Initial guess
        # -----------------------------------------------------

        initial_guess = np.array([
            np.deg2rad(alpha_guess_deg),
            np.deg2rad(elevator_guess_deg),
            throttle_guess,
        ])

        # -----------------------------------------------------
        # Bounds
        # -----------------------------------------------------

        lower_bounds = np.array([
            np.deg2rad(-5.0),    # alpha
            np.deg2rad(-25.0),   # elevator
            0.0,                 # throttle
        ])

        upper_bounds = np.array([
            np.deg2rad(15.0),    # alpha
            np.deg2rad(25.0),    # elevator
            1.0,                 # throttle
        ])

        # -----------------------------------------------------
        # Nonlinear least-squares solver
        # -----------------------------------------------------

        result = least_squares(
            fun=self.residual,
            x0=initial_guess,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            args=(
                airspeed,
                altitude,
            ),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=1000,
        )

        # -----------------------------------------------------
        # Extract solution
        # -----------------------------------------------------

        alpha = result.x[0]
        elevator = result.x[1]
        throttle = result.x[2]

        theta = alpha

        state = self.build_state(
            airspeed=airspeed,
            altitude=altitude,
            alpha=alpha,
        )

        controls = self.build_controls(
            elevator=elevator,
            throttle=throttle,
        )

        residual = self.residual(
            result.x,
            airspeed,
            altitude,
        )

        converged = bool(
            result.success
            and np.all(
                np.isfinite(residual)
            )
        )

        return TrimSolution(
            airspeed=airspeed,
            altitude=altitude,

            alpha=alpha,
            theta=theta,

            elevator=elevator,
            throttle=throttle,

            residual=residual,

            converged=converged,

            cost=float(
                result.cost
            ),

            optimality=float(
                result.optimality
            ),

            state=state,
            controls=controls,
        )