import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

SUPPORTED_EXTENSIONS: Set[str] = {".csv", ".xlsx", ".json", ".parquet"}


class FileFormatError(Exception):
    """Raised when an input file is missing, unsupported, or fails format/hash checks."""
    pass


def load_configuration(config_path: str | Path = "config.json") -> Dict[str, Any]:
    """Load config.json and attach the parsed manifest.json dictionary."""
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    manifest_path = Path(config.get("manifest_path", "data/manifest.json"))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    config["manifest"] = manifest
    config["file_specs"] = {
        entry["file"]: entry for entry in manifest.get("files", [])
    }
    return config


def compute_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file for manifest verification."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_input_file(
    file_path: str | Path,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> Path:
    """Verify that the input file exists, has a supported extension, and matches size/hash."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileFormatError(
            f"Unsupported file format '{ext}' for '{path.name}'."
        )

    if expected_bytes is not None:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise FileFormatError(f"Byte size mismatch for {path.name}.")

    if expected_sha256 is not None:
        actual_sha256 = compute_sha256(path)
        if actual_sha256 != expected_sha256:
            raise FileFormatError(f"SHA-256 mismatch for {path.name}.")

    return path


def read_csv_records(
    file_path: str | Path,
    encoding: str = "cp1252",
    separator: str = ";",
    comment_prefix: Optional[str] = "#",
    ignore_columns: Optional[List[str]] = None,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> pd.DataFrame:
    """Read a CSV input file preserving raw string values and 1-based source_record order."""
    path = validate_input_file(
        file_path, expected_sha256=expected_sha256, expected_bytes=expected_bytes
    )
    if path.suffix.lower() != ".csv":
        raise FileFormatError(f"Expected a .csv file, got: {path.name}")

    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=separator,
            comment=comment_prefix,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise FileFormatError(f"Failed to parse CSV '{path.name}': {exc}") from exc

    if ignore_columns:
        cols_to_drop = [c for c in ignore_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    df.insert(0, "source_record", range(1, len(df) + 1))
    df.insert(0, "source_file", path.name)
    return df


def read_excel_records(
    file_path: str | Path,
    sheet_name: str = "observations",
    engine: str = "openpyxl",
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> pd.DataFrame:
    """Read country_records.xlsx from 'observations' sheet preserving raw strings and 1-based order."""
    path = validate_input_file(
        file_path, expected_sha256=expected_sha256, expected_bytes=expected_bytes
    )
    if path.suffix.lower() != ".xlsx":
        raise FileFormatError(f"Expected an .xlsx file, got: {path.name}")

    try:
        excel_file = pd.ExcelFile(path, engine=engine)
        if sheet_name not in excel_file.sheet_names:
            raise FileFormatError(f"Sheet '{sheet_name}' not found.")

        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False,
        )
    except FileFormatError:
        raise
    except Exception as exc:
        raise FileFormatError(f"Failed to parse Excel '{path.name}': {exc}") from exc

    df.insert(0, "source_record", range(1, len(df) + 1))
    df.insert(0, "source_file", path.name)
    return df