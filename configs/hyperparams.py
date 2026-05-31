"""Load tuned MSSD/HNMD hyperparameters from bundled Excel tables."""

from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from configs.datasets import MSSD_GRID_EXPONENTS, MSSD_GRID_KNOT_MULTIPLIERS


@dataclass(frozen=True)
class MSSDHyperparams:
    alpha: float
    beta: float
    gamma: float
    knot_multiplier: float
    spline_criterion_exponent: int
    max_levels: int = 5


# Fallback values from the paper's Table 4 / search_greek.py when Excel rows are missing.
_FALLBACK_MLP: Dict[Tuple[str, int], MSSDHyperparams] = {}
_rows = {
    "ETTm1": [(96, 0.01, 5, 1, 5, 1), (192, 0.01, 5, 1, 5, 1), (336, 0.01, 5, 1, 5, 1), (720, 0.01, 5, 1, 5, 1)],
    "ETTm2": [(96, 0.1, 1, 1, 3, 2), (192, 0.01, 1, 1, 5, 1), (336, 0.1, 1, 5, 5, 1), (720, 0.01, 1, 5, 5, 1)],
    "ETTh1": [(96, 0.01, 2, 1, 3, 1), (192, 0.01, 2, 2, 3, 1), (336, 0.01, 2, 3, 5, 1), (720, 0.01, 2, 3, 5, 1)],
    "ETTh2": [(96, 1.0, 0, 1, 5, 5), (192, 0.1, 1, 1, 5, 5), (336, 0.1, 0, 0, 5, 5), (720, 0.1, 2, 3, 5, 2)],
    "electricity": [(96, 0.01, 0, 5, 3, 2), (192, 0.01, 2, 2, 1, 1), (336, 0.01, 2, 2, 5, 3), (720, 0.01, 2, 5, 5, 1)],
    "exchange_rate": [(96, 0.01, 1, 1, 5, 5), (192, 0.01, 0, 3, 5, 5), (336, 0.1, 0, 2, 5, 3), (720, 0.1, 0, 3, 5, 4)],
    "traffic": [(96, 0.01, 0, 3, 5, 5), (192, 0.01, 0, 3, 5, 5), (336, 0.01, 0, 3, 5, 5), (720, 0.01, 0, 5, 5, 1)],
    "weather": [(96, 0.01, 1, 1, 3, 5), (192, 0.01, 0, 5, 5, 5), (336, 0.01, 0, 3, 5, 2), (720, 0.01, 0, 5, 5, 5)],
    "Solar": [(96, 0.01, 2, 2, 3, 2), (192, 0.01, 2, 2, 3, 2), (336, 0.01, 2, 2, 5, 3), (720, 0.01, 2, 5, 5, 3)],
}
for dataset, entries in _rows.items():
    for pred_len, alpha, beta, gamma, ksf, sce in entries:
        _FALLBACK_MLP[(dataset, pred_len)] = MSSDHyperparams(
            alpha=alpha,
            beta=float(beta),
            gamma=float(gamma),
            knot_multiplier=float(ksf),
            spline_criterion_exponent=int(sce),
        )

_EXCEL_BY_MODEL = {
    "MLP": "HNMV_MLP_Hyperparam_testing_optimized.xlsx",
    "DLinear": "HNMV_DLinear_optimized.xlsx",
}

_EXCEL_CACHE: Dict[str, Dict[Tuple[str, int], MSSDHyperparams]] = {}


def _parse_prefixed_float(value: str) -> float:
    """Parse values like 'a0.01', 'b2.0', 'g1.0', or plain floats."""
    if value is None:
        raise ValueError("empty hyperparameter value")
    text = str(value).strip()
    if not text:
        raise ValueError("empty hyperparameter value")
    match = re.match(r"^[abg](.+)$", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1)
    return float(text)


def _read_xlsx_rows(path: str):
    with zipfile.ZipFile(path) as zf:
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for row in sheet.findall(".//m:row", ns):
        vals = []
        for cell in row.findall("m:c", ns):
            inline = cell.find("m:is", ns)
            if inline is not None:
                text = inline.find("m:t", ns)
                vals.append(text.text if text is not None else "")
                continue
            value = cell.find("m:v", ns)
            vals.append(value.text if value is not None else "")
        rows.append(vals)
    return rows


def _load_excel_hyperparams(model: str) -> Dict[Tuple[str, int], MSSDHyperparams]:
    if model in _EXCEL_CACHE:
        return _EXCEL_CACHE[model]

    filename = _EXCEL_BY_MODEL.get(model)
    if filename is None or not os.path.exists(filename):
        _EXCEL_CACHE[model] = {}
        return _EXCEL_CACHE[model]

    rows = _read_xlsx_rows(filename)
    header = rows[0]
    idx = {name: header.index(name) for name in header}

    parsed: Dict[Tuple[str, int], MSSDHyperparams] = {}
    for row in rows[1:]:
        if len(row) <= max(idx.values()):
            continue
        feature = row[idx["feature"]]
        pred_len = int(float(row[idx["pred_len"]]))
        beta_col = "bets" if "bets" in idx else "beta"
        parsed[(feature, pred_len)] = MSSDHyperparams(
            alpha=_parse_prefixed_float(row[idx["alpha"]]),
            beta=_parse_prefixed_float(row[beta_col]),
            gamma=_parse_prefixed_float(row[idx["gamma"]]),
            knot_multiplier=float(row[idx["knot_multiplier"]]),
            spline_criterion_exponent=int(float(row[idx["spline_criterion_exponent"]])),
        )

    _EXCEL_CACHE[model] = parsed
    return parsed


def get_mssd_hyperparams(model: str, dataset: str, pred_len: int) -> MSSDHyperparams:
    excel = _load_excel_hyperparams(model)
    key = (dataset, pred_len)
    if key in excel:
        return excel[key]

    if model not in _EXCEL_BY_MODEL:
        mlp_excel = _load_excel_hyperparams("MLP")
        if key in mlp_excel:
            return mlp_excel[key]

    if key in _FALLBACK_MLP:
        return _FALLBACK_MLP[key]
    raise KeyError(f"No MSSD hyperparameters for model={model}, dataset={dataset}, pred_len={pred_len}")


def iter_mssd_grid_search():
    """Yield knot_multiplier / exponent pairs for grid-search datasets."""
    for knot_multiplier in MSSD_GRID_KNOT_MULTIPLIERS:
        for exponent in MSSD_GRID_EXPONENTS:
            yield knot_multiplier, exponent
