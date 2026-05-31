"""Grid-search MSSD hyperparameters (knot multiplier x spline exponent)."""

from run_experiments import main

if __name__ == "__main__":
    import sys

    sys.argv = [
        "experimentation.py",
        "--models",
        "MLP",
        "DLinear",
        "SOFTS",
        "--datasets",
        "all",
        "--losses",
        "mssd",
        "--mssd_mode",
        "grid",
        "--itr",
        "1",
        "--seeds",
        "2024",
    ] + sys.argv[1:]
    main()
