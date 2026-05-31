import os

import pandas as pd

from utils.results_parser import collect_results


if __name__ == "__main__":
    base_dir = os.getcwd()
    df = collect_results(os.path.join(base_dir, "results"))
    output_path = os.path.join(base_dir, "output.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Saved {len(df)} result rows to {output_path}")
