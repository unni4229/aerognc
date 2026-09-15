import numpy as np

from dynamics.kinematics import body_to_ned


def test_yaw_0():

    body_velocity = np.array([
        25.0,
        0.0,
        0.0,
    ])

    ned = body_to_ned(
        body_velocity,
        0.0,
        0.0,
        0.0,
    )

    expected = np.array([
        25.0,
        0.0,
        0.0,
    ])

    assert np.allclose(
        ned,
        expected,
        atol=1e-10,
    )


def test_yaw_90():

    body_velocity = np.array([
        25.0,
        0.0,
        0.0,
    ])

    ned = body_to_ned(
        body_velocity,
        0.0,
        0.0,
        np.deg2rad(90.0),
    )

    expected = np.array([
        0.0,
        25.0,
        0.0,
    ])

    assert np.allclose(
        ned,
        expected,
        atol=1e-10,
    )


def test_yaw_180():

    body_velocity = np.array([
        25.0,
        0.0,
        0.0,
    ])

    ned = body_to_ned(
        body_velocity,
        0.0,
        0.0,
        np.deg2rad(180.0),
    )

    expected = np.array([
        -25.0,
        0.0,
        0.0,
    ])

    assert np.allclose(
        ned,
        expected,
        atol=1e-10,
    )


if __name__ == "__main__":

    test_yaw_0()
    test_yaw_90()
    test_yaw_180()

    print("Rotation tests PASSED.")