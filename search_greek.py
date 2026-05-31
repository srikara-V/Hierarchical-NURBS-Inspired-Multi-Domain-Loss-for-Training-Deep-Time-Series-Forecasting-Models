"""Run tuned MSSD/HNMD experiments using optimized hyperparameter tables."""

from run_experiments import main

if __name__ == "__main__":
    import sys

    sys.argv = [
        "search_greek.py",
        "--models",
        "MLP",
        "--datasets",
        "all",
        "--losses",
        "mssd",
        "--mssd_mode",
        "tuned",
        "--itr",
        "1",
        "--seeds",
        "2024",
    ] + sys.argv[1:]
    main()
