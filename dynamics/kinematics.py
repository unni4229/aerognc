import numpy as np


def body_to_ned_dcm(
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Direction Cosine Matrix (DCM) that transforms
    a vector from body coordinates to NED coordinates.

    v_ned = C_bn @ v_body
    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    C_bn = np.array([
        [
            ctheta * cpsi,
            sphi * stheta * cpsi
            - cphi * spsi,
            cphi * stheta * cpsi
            + sphi * spsi,
        ],

        [
            ctheta * spsi,
            sphi * stheta * spsi
            + cphi * cpsi,
            cphi * stheta * spsi
            - sphi * cpsi,
        ],

        [
            -stheta,
            sphi * ctheta,
            cphi * ctheta,
        ],
    ])

    return C_bn


def ned_to_body_dcm(
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Direction Cosine Matrix that transforms
    a vector from NED coordinates to body coordinates.

    v_body = C_nb @ v_ned
    """

    C_bn = body_to_ned_dcm(
        phi,
        theta,
        psi,
    )

    return C_bn.T


def body_to_ned(
    vector_body: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Transform a 3-element body-frame vector
    into NED coordinates.
    """

    vector_body = np.asarray(
        vector_body,
        dtype=float,
    )

    if vector_body.shape != (3,):
        raise ValueError(
            "Input vector must have shape (3,)."
        )

    C_bn = body_to_ned_dcm(
        phi,
        theta,
        psi,
    )

    return C_bn @ vector_body


def ned_to_body(
    vector_ned: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Transform a 3-element NED-frame vector
    into body coordinates.
    """

    vector_ned = np.asarray(
        vector_ned,
        dtype=float,
    )

    if vector_ned.shape != (3,):
        raise ValueError(
            "Input vector must have shape (3,)."
        )

    C_nb = ned_to_body_dcm(
        phi,
        theta,
        psi,
    )

    return C_nb @ vector_ned


def euler_rate_matrix(
    phi: float,
    theta: float,
) -> np.ndarray:
    """
    Transformation from body angular rates [p,q,r]
    to Euler angle rates [phi_dot, theta_dot, psi_dot].
    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    ctheta = np.cos(theta)

    if abs(ctheta) < 1e-6:
        raise ValueError(
            "Euler-angle singularity approaching "
            "pitch = +/-90 degrees."
        )

    ttheta = np.tan(theta)

    return np.array([
        [
            1.0,
            sphi * ttheta,
            cphi * ttheta,
        ],

        [
            0.0,
            cphi,
            -sphi,
        ],

        [
            0.0,
            sphi / ctheta,
            cphi / ctheta,
        ],
    ])