from dataclasses import dataclass

import numpy as np

from models.aircraft import AircraftParameters
from models.controls import ControlInput
from models.aircraft_dynamics import aircraft_derivatives

from simulation.integrator import rk4_step


@dataclass
class SimulationSample:
    """
    One simulation record.
    """

    time: float
    state: np.ndarray
    controls: ControlInput


class AircraftSimulator:
    """
    Nonlinear fixed-wing aircraft simulation wrapper.

    The simulator:
        1. stores the current aircraft state,
        2. evaluates the aircraft dynamics,
        3. integrates the state using RK4,
        4. logs the result.
    """

    def __init__(
        self,
        params: AircraftParameters,
        dt: float = 0.01,
    ):

        if dt <= 0.0:
            raise ValueError(
                "Simulation timestep must be positive."
            )

        self.params = params
        self.dt = dt

        self.state = np.zeros(
            12,
            dtype=float,
        )

        self.time = 0.0

        self.history: list[SimulationSample] = []

    def reset(
        self,
        state: np.ndarray,
    ) -> None:
        """
        Reset the simulator to a specified state.
        """

        state = np.asarray(
            state,
            dtype=float,
        )

        if state.shape != (12,):
            raise ValueError(
                "Initial aircraft state must "
                "contain exactly 12 elements."
            )

        if not np.all(
            np.isfinite(state)
        ):
            raise ValueError(
                "Initial aircraft state contains "
                "non-finite values."
            )

        self.state = state.copy()

        self.time = 0.0

        self.history.clear()

    def step(
        self,
        controls: ControlInput,
    ) -> np.ndarray:
        """
        Propagate the aircraft by one timestep.
        """

        controls = controls.clipped()

        def derivative(
            current_state: np.ndarray,
        ) -> np.ndarray:

            return aircraft_derivatives(
                state=current_state,
                controls=controls,
                params=self.params,
            )

        self.state = rk4_step(
            derivative_function=derivative,
            state=self.state,
            dt=self.dt,
        )

        self.time += self.dt

        if not np.all(
            np.isfinite(self.state)
        ):
            raise RuntimeError(
                "Aircraft simulation produced "
                "non-finite state values."
            )

        self.history.append(
            SimulationSample(
                time=self.time,
                state=self.state.copy(),
                controls=controls,
            )
        )

        return self.state.copy()

    def run(
        self,
        simulation_time: float,
        controls: ControlInput,
    ) -> list[SimulationSample]:
        """
        Run an open-loop simulation using constant controls.
        """

        if simulation_time <= 0.0:
            raise ValueError(
                "Simulation time must be positive."
            )

        steps = int(
            np.ceil(
                simulation_time / self.dt
            )
        )

        for _ in range(steps):
            self.step(controls)

        return self.history