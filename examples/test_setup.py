import numpy as np
import scipy
import matplotlib
import pandas
import yaml


def main():
    print("=" * 40)
    print("AeroGNC environment is working!")
    print("=" * 40)

    print(f"NumPy      : {np.__version__}")
    print(f"SciPy      : {scipy.__version__}")
    print(f"Matplotlib : {matplotlib.__version__}")
    print(f"Pandas     : {pandas.__version__}")
    print("PyYAML     : imported successfully")


if __name__ == "__main__":
    main()