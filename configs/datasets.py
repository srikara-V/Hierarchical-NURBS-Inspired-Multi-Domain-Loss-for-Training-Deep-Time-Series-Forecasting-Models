"""Benchmark dataset definitions for long-horizon forecasting experiments."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root_path: str
    data_path: str
    data: str
    enc_in: int
    freq: str = "h"
    batch_size: int = 32
    learning_rate: float = 0.001
    train_epochs: int = 10
    patience: int = 3


DATASETS: Dict[str, DatasetConfig] = {
    "ETTm1": DatasetConfig(
        name="ETTm1",
        root_path="./data/ETT-small/",
        data_path="ETTm1.csv",
        data="ETTm1",
        enc_in=7,
        freq="min",
    ),
    "ETTm2": DatasetConfig(
        name="ETTm2",
        root_path="./data/ETT-small/",
        data_path="ETTm2.csv",
        data="ETTm2",
        enc_in=7,
        freq="min",
    ),
    "ETTh1": DatasetConfig(
        name="ETTh1",
        root_path="./data/ETT-small/",
        data_path="ETTh1.csv",
        data="ETTh1",
        enc_in=7,
        freq="h",
    ),
    "ETTh2": DatasetConfig(
        name="ETTh2",
        root_path="./data/ETT-small/",
        data_path="ETTh2.csv",
        data="ETTh2",
        enc_in=7,
        freq="h",
    ),
    "electricity": DatasetConfig(
        name="electricity",
        root_path="./data/electricity/",
        data_path="electricity.csv",
        data="custom",
        enc_in=321,
        batch_size=16,
        learning_rate=0.0005,
    ),
    "exchange_rate": DatasetConfig(
        name="exchange_rate",
        root_path="./data/exchange_rate/",
        data_path="exchange_rate.csv",
        data="custom",
        enc_in=8,
    ),
    "weather": DatasetConfig(
        name="weather",
        root_path="./data/weather/",
        data_path="weather.csv",
        data="custom",
        enc_in=21,
    ),
    "traffic": DatasetConfig(
        name="traffic",
        root_path="./data/traffic/",
        data_path="traffic.csv",
        data="custom",
        enc_in=862,
        batch_size=16,
        learning_rate=0.001,
    ),
    "Solar": DatasetConfig(
        name="Solar",
        root_path="./data/Solar/",
        data_path="solar_AL.txt",
        data="Solar",
        enc_in=137,
        learning_rate=0.0005,
    ),
}

PRED_LENS: List[int] = [96, 192, 336, 720]

# Datasets where MSSD hyperparameters were selected via grid search rather than
# the optimized Excel tables.
MSSD_GRID_SEARCH_DATASETS = {"ETTm1", "ETTm2", "electricity"}

MSSD_GRID_KNOT_MULTIPLIERS = [1, 2, 3, 5]
MSSD_GRID_EXPONENTS = [1, 2, 5, 8]


def get_dataset(name: str) -> DatasetConfig:
    key = name.strip()
    aliases = {
        "ECL": "electricity",
        "exchange": "exchange_rate",
        "solar": "Solar",
        "solar-energy": "Solar",
        "Solar-Energy": "Solar",
    }
    key = aliases.get(key, key)
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}")
    return DATASETS[key]


def list_datasets(names: Optional[List[str]] = None) -> List[DatasetConfig]:
    if not names or names == ["all"]:
        return list(DATASETS.values())
    return [get_dataset(name) for name in names]
