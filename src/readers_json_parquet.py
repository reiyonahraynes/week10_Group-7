import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.readers import FileFormatError, validate_input_file


def read_json_records(file_path: str | Path, records_key: str = "records", expected_sha256: Optional[str] = None, expected_bytes: Optional[int] = None,) -> pd.DataFrame:
    path = validate_input_file(file_path, expected_sha256 = expected_sha256, expected_bytes = expected_bytes)
    if path.suffix.lower() != ".json":
        raise FileFormatError(f"Expected a .json file, got: {path.name}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and records_key in data:
            records = data[records_key]
        elif isinstance(data, list):
            records = data
        else:
            raise FileFormatError(f"JSON structure does not contain records list under '{records_key}'")

        df = pd.json_normalize(records)
        df = df.astype(str)
    except FileFormatError:
        raise
    except Exception as exc:
        raise FileFormatError(f"Failed to parse JSON '{path.name}': {exc}") from exc

    df.insert(0, "source_record", range(1, len(df) + 1))
    df.insert(0, "source_file", path.name)
    return df

def read_parquet_records(file_path: str | Path, expected_sha256: Optional[str] = None, expected_bytes: Optional[int] = None,) -> pd.DataFrame:
    path = validate_input_file(file_path, expected_sha256 = expected_sha256, expected_bytes = expected_bytes)
    if path.suffix.lower() != ".parquet":
        raise FileFormatError(f"Expected a .parquet file, got: {path.name}")

    try:
        df = pd.read_parquet(path)
        df = df.astype(str)
    except Exception as exc:
        raise FileFormatError(f"Failed to parse Parquet '{path.name}': {exc}") from exc

    df.insert(0, "source_record", range(1, len(df) + 1))
    df.insert(0, "source_file", path.name)
    return df

def map_canonical_columns(df: pd.DataFrame, column_mapping: Dict[str, str], required_columns: List[str],) -> pd.DataFrame:
    mapped_df = df.rename(columns = column_mapping)
    missing = [column for column in required_columns if column not in mapped_df.columns]
    if missing:
        raise FileFormatError(f"Missing required canonical column(s): {missing}")
    
    canonical_order = ["source_file", "source_record"] + required_columns
    extra_columns = [c for c in mapped_df.columns if c not in canonical_order]
    return mapped_df[canonical_order + extra_columns]
