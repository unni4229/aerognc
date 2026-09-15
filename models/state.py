from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftState:
    """
    12-state representation of a rigid-body fixed-wing aircraft.

    Position:
        pn, pe, pd     [m]

    Body velocity:
        u, v, w        [m/s]

    Euler attitude:
        phi, theta, psi [rad]

    Body angular rates:
        p, q, r        [rad/s]

    Coordinate convention:
        Navigation frame = NED
        N = North
        E = East
        D = Down

        Body frame:
        x = forward
        y = right
        z = down
    """

    # Position in NED
    pn: float
    pe: float
    pd: float

    # Velocity in body frame
    u: float
    v: float
    w: float

    # Euler attitude
    phi: float
    theta: float
    psi: float

    # Body angular rates
    p: float
    q: float
    r: float

    def as_vector(self) -> np.ndarray:
        """
        Convert the aircraft state into a 12x1 NumPy vector.
        """

        return np.array([
            self.pn,
            self.pe,
            self.pd,

            self.u,
            self.v,
            self.w,

            self.phi,
            self.theta,
            self.psi,

            self.p,
            self.q,
            self.r,
        ], dtype=float)

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
    ) -> "AircraftState":
        """
        Create an AircraftState object from a 12-element vector.
        """

        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.shape != (12,):
            raise ValueError(
                "Aircraft state vector must "
                "contain exactly 12 elements."
            )

        return cls(
            pn=vector[0],
            pe=vector[1],
            pd=vector[2],

            u=vector[3],
            v=vector[4],
            w=vector[5],

            phi=vector[6],
            theta=vector[7],
            psi=vector[8],

            p=vector[9],
            q=vector[10],
            r=vector[11],
        )

    @property
    def altitude(self) -> float:
        """
        Altitude above the NED reference origin.

        Since Down is positive:
            altitude = -pd
        """

        return -self.pd

    @property
    def airspeed(self) -> float:
        """
        Body-frame airspeed magnitude.
        """

        return float(
            np.sqrt(
                self.u**2
                + self.v**2
                + self.w**2
            )
        )