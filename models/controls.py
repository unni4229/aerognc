from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlInput:
    """
    Aircraft control inputs.

    Aileron, elevator, rudder:
        radians

    Throttle:
        normalized 0...1
    """

    aileron: float
    elevator: float
    rudder: float
    throttle: float

    def clipped(self) -> "ControlInput":

        return ControlInput(
            aileron=float(
                np.clip(
                    self.aileron,
                    np.deg2rad(-25.0),
                    np.deg2rad(25.0),
                )
            ),

            elevator=float(
                np.clip(
                    self.elevator,
                    np.deg2rad(-25.0),
                    np.deg2rad(25.0),
                )
            ),

            rudder=float(
                np.clip(
                    self.rudder,
                    np.deg2rad(-30.0),
                    np.deg2rad(30.0),
                )
            ),

            throttle=float(
                np.clip(
                    self.throttle,
                    0.0,
                    1.0,
                )
            ),
        )