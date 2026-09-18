from models.atmosphere import air_density


def main():

    print("=" * 50)
    print("AEROGNC ATMOSPHERE TEST")
    print("=" * 50)

    altitudes = [
        0.0,
        100.0,
        500.0,
        1000.0,
    ]

    for altitude in altitudes:

        rho = air_density(altitude)

        print(
            f"Altitude = {altitude:7.1f} m"
            f"   Density = {rho:.6f} kg/m^3"
        )

    print("\nAtmosphere test complete.")


if __name__ == "__main__":
    main()