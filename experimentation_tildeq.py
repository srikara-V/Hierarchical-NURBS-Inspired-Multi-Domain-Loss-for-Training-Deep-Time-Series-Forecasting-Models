"""Run Tilde-Q baseline experiments across the 9-dataset benchmark."""

from run_experiments import main

if __name__ == "__main__":
    import sys

    sys.argv = [
        "experimentation_tildeq.py",
        "--models",
        "MLP",
        "DLinear",
        "SOFTS",
        "iTransformer",
        "--datasets",
        "all",
        "--losses",
        "tildeq",
        "--itr",
        "3",
        "--seeds",
        "2024",
    ] + sys.argv[1:]
    main()
