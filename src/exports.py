
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from src.compare import CANONICAL_ORDER, apply_schema, compare_tables, schema_dtypes

SHEET_NAME = "observations"
EXPECTED_KINDS = {"country": "text", "continent": "text", "year": "int64",
                  "lifeExp": "float64", "pop": "int64", "gdpPercap": "float64"}


def _native_records(table: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return rows as plain Python values so json.dump keeps exact numbers."""
    records = []
    for row in table.to_dict("records"):
        records.append({"country": str(row["country"]), "continent": str(row["continent"]),
                        "year": int(row["year"]), "lifeExp": float(row["lifeExp"]),
                        "pop": int(row["pop"]), "gdpPercap": float(row["gdpPercap"])})
    return records


def export_csv(table: pd.DataFrame, path: Path) -> None:
    """Write UTF-8, comma separated, with header and no index."""
    table.to_csv(path, index=False, encoding="utf-8")


def export_json(table: pd.DataFrame, path: Path) -> None:
    """Write {"records": [...]} as UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump({"records": _native_records(table)}, f, ensure_ascii=False, indent=1)


def export_parquet(table: pd.DataFrame, path: Path) -> None:
    """Write Parquet with the pyarrow engine."""
    table.to_parquet(path, engine="pyarrow", index=False)


def export_xlsx(table: pd.DataFrame, path: Path) -> None:
    """Write one sheet called 'observations' with the openpyxl engine."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=SHEET_NAME, index=False)


def reload_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False)


def reload_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)["records"])


def reload_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def reload_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl", keep_default_na=False)


EXPORTERS: Dict[str, Tuple[str, Callable, Callable]] = {
    "csv": ("canonical_dataset.csv", export_csv, reload_csv),
    "json": ("canonical_dataset.json", export_json, reload_json),
    "parquet": ("canonical_dataset.parquet", export_parquet, reload_parquet),
    "xlsx": ("canonical_dataset.xlsx", export_xlsx, reload_xlsx),
}


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_and_verify(table: pd.DataFrame, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Write all four exports to out_dir, reload each, and compare to table.

    Returns (check_rows, export_info). A reload problem becomes a failed check
    rather than an exception, so the run can report every format at once.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = apply_schema(table)
    checks: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []

    def row(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"check": name, "expected": 1, "actual": int(ok), "tolerance": 0.0,
                       "pass": bool(ok), "detail": str(detail)})

    for fmt, (filename, writer, reader) in EXPORTERS.items():
        path = out_dir / filename
        try:
            writer(expected, path)
            reloaded = apply_schema(reader(path))
        except Exception as exc:  # any failure to write or read back is a failed check
            row(f"export_{fmt}_reload", False, f"{type(exc).__name__}: {exc}")
            continue
        report = compare_tables(expected, reloaded, "written", "reloaded")
        row(f"export_{fmt}_rows", len(reloaded) == len(expected), f"{len(reloaded)} vs {len(expected)}")
        row(f"export_{fmt}_values", report["agree"],
            f"{len(report['mismatches'])} mismatches, {len(report['only_left'])} missing, "
            f"{len(report['only_right'])} extra")
        row(f"export_{fmt}_dtypes", schema_dtypes(reloaded) == EXPECTED_KINDS, schema_dtypes(reloaded))
        row(f"export_{fmt}_columns", list(reloaded.columns) == CANONICAL_ORDER, list(reloaded.columns))
        info.append({"format": fmt, "file": filename, "bytes": path.stat().st_size, "sha256": sha256_of(path)})
    return checks, info
