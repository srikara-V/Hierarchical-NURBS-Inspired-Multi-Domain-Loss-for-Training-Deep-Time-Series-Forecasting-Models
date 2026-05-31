"""Run tuned MSSD/HNMD experiments on the Traffic dataset."""

from run_experiments import main

if __name__ == "__main__":
    import sys

    sys.argv = [
        "experimentation_mssdtraffic.py",
        "--models",
        "MLP",
        "--datasets",
        "traffic",
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
