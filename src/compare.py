from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CANONICAL_ORDER: List[str] = ["country", "continent", "year", "lifeExp", "pop", "gdpPercap"]
KEY: List[str] = ["country", "year"]
TEXT_COLUMNS = ("country", "continent")
INT_COLUMNS = ("year", "pop")
FLOAT_COLUMNS = ("lifeExp", "gdpPercap")
FLOAT_ABS_TOL: float = 1e-9


class SchemaError(Exception):
    pass


class ComparisonError(Exception):
    pass


def apply_schema(df: pd.DataFrame, order: Sequence[str] = CANONICAL_ORDER) -> pd.DataFrame:
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise SchemaError(f"missing canonical columns: {missing}")
    out = df[list(order)].copy()
    try:
        for col in TEXT_COLUMNS:
            if out[col].isna().any():
                raise SchemaError(f"missing values in text column {col}")
            out[col] = out[col].astype(str).str.strip()
        for col in INT_COLUMNS:
            numbers = pd.to_numeric(out[col], errors="raise")
            if numbers.isna().any() or not np.all(np.mod(numbers.to_numpy(dtype=float), 1) == 0):
                raise SchemaError(f"column {col} is not integral")
            out[col] = numbers.astype("int64")
        for col in FLOAT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"cannot apply schema: {exc}") from exc
    return out.sort_values(KEY).reset_index(drop=True)


def schema_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            kinds[col] = "int64"
        elif pd.api.types.is_float_dtype(df[col]):
            kinds[col] = "float64"
        else:
            kinds[col] = "text"
    return kinds


def compare_tables(left: pd.DataFrame, right: pd.DataFrame, left_name: str = "left",
                   right_name: str = "right", float_tol: float = FLOAT_ABS_TOL) -> Dict[str, Any]:
    for name, table in ((left_name, left), (right_name, right)):
        if table.duplicated(KEY).any():
            raise ComparisonError(f"{name} has duplicate (country, year) keys")
    merged = left.merge(right, on=KEY, how="outer", suffixes=("_l", "_r"), indicator=True)
    only_left = merged[merged["_merge"] == "left_only"][KEY]
    only_right = merged[merged["_merge"] == "right_only"][KEY]
    both = merged[merged["_merge"] == "both"]

    mismatches: List[Dict[str, Any]] = []
    for col in CANONICAL_ORDER:
        if col in KEY:
            continue
        a, b = both[f"{col}_l"], both[f"{col}_r"]
        if col in FLOAT_COLUMNS:
            bad = ~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0.0, atol=float_tol)
        else:
            bad = (a != b).to_numpy()
        for pos in np.flatnonzero(bad):
            row = both.iloc[pos]
            mismatches.append({"country": row["country"], "year": int(row["year"]), "column": col,
                               left_name: row[f"{col}_l"], right_name: row[f"{col}_r"]})
    return {
        "left": left_name, "right": right_name,
        "rows_left": len(left), "rows_right": len(right),
        "only_left": only_left.to_dict("records"), "only_right": only_right.to_dict("records"),
        "mismatches": mismatches,
        "agree": bool(only_left.empty and only_right.empty and not mismatches),
    }


def compare_all(tables: Dict[str, pd.DataFrame], float_tol: float = FLOAT_ABS_TOL) -> pd.DataFrame:
    if len(tables) < 2:
        raise ComparisonError("need at least two tables to compare")
    names = list(tables)
    reference = names[0]
    rows = []
    for other in names[1:]:
        rep = compare_tables(tables[reference], tables[other], reference, other, float_tol)
        rows.append({"reference": reference, "compared": other,
                     "rows_reference": rep["rows_left"], "rows_compared": rep["rows_right"],
                     "only_in_reference": len(rep["only_left"]), "only_in_compared": len(rep["only_right"]),
                     "value_mismatches": len(rep["mismatches"]), "agree": rep["agree"]})
    return pd.DataFrame(rows)