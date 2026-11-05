"""Row validation, quarantine, duplicate and conflict handling"""
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

CANONICAL_COLUMNS: List[str] = ["country", "continent", "year", "lifeExp", "pop", "gdpPercap"]
TEXT_FIELDS: Set[str] = {"country", "continent"}
MISSING_TEXT: Set[str] = {"", "none", "nan", "null"}
NON_FINITE_TEXT: Set[str] = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
NUMBER_PATTERN = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
MAX_LIFE_EXPECTANCY = 120.0


class ConflictingKeyError(Exception):
    """Raised when two valid rows share (country, year) but disagree elsewhere."""


def is_missing(value: Any) -> bool:
    """Return True for None, NaN, blank text, or the strings 'None'/'nan'."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in MISSING_TEXT


def parse_number(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse a value to float; return (number, None) or (None, 'missing'|'invalid'|'non-finite')."""
    if is_missing(value):
        return None, "missing"
    if isinstance(value, (bool, np.bool_)):
        return None, "invalid"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in NON_FINITE_TEXT:
            return None, "non-finite"
        if not NUMBER_PATTERN.match(text):  # rejects '1,000', '1_0', '0x10'
            return None, "invalid"
        number = float(text)
    else:
        return None, "invalid"
    if not math.isfinite(number):
        return None, "non-finite"
    return number, None


def build_rules(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Build validation rules (sets of allowed values and numeric bounds) from the manifest."""
    return {"continents": set(manifest["allowed_continents"]),
            "years": {int(y) for y in manifest["allowed_years"]},
            "max_life_exp": MAX_LIFE_EXPECTANCY}


def whole_number(number: float) -> Optional[int]:
    """Return the integer value of a float, or None if it has a fractional part."""
    return int(number) if float(number).is_integer() else None


def validate_record(record: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Validate one record; return (reasons, typed row). Typed row is empty when rejected."""
    reasons: List[str] = []
    typed: Dict[str, Any] = {}

    for field in TEXT_FIELDS:
        value = record.get(field)
        if is_missing(value):
            reasons.append(f"missing {field}")
        else:
            typed[field] = str(value).strip()
    if "continent" in typed and typed["continent"] not in rules["continents"]:
        reasons.append(f"continent not allowed: {typed['continent']}")

    numbers: Dict[str, float] = {}
    for field in ("year", "lifeExp", "pop", "gdpPercap"):
        number, problem = parse_number(record.get(field))
        if problem:
            reasons.append(f"{problem} {field}")
        else:
            numbers[field] = number  # type: ignore[assignment]

    for field in ("year", "pop"):
        if field in numbers:
            whole = whole_number(numbers[field])
            if whole is None:
                reasons.append(f"fractional {field}")
                del numbers[field]
            else:
                typed[field] = whole
    if "year" in typed and typed["year"] not in rules["years"]:
        reasons.append(f"year not allowed: {typed['year']}")
    if "pop" in typed and typed["pop"] <= 0:
        reasons.append("pop must be positive")
    if "lifeExp" in numbers:
        if not 0 < numbers["lifeExp"] <= rules["max_life_exp"]:
            reasons.append("lifeExp out of range")
        typed["lifeExp"] = numbers["lifeExp"]
    if "gdpPercap" in numbers:
        if numbers["gdpPercap"] <= 0:
            reasons.append("gdpPercap must be positive")
        typed["gdpPercap"] = numbers["gdpPercap"]

    if reasons:
        return reasons, {}
    return [], {c: typed[c] for c in CANONICAL_COLUMNS}


class ValidationRun:
    """Tracks quarantined rows, removed duplicates and counts for one validation run."""

    def __init__(self, rules: Dict[str, Any]) -> None:
        """Store the rules and start with empty logs."""
        self.rules = rules
        self.quarantine: List[Dict[str, Any]] = []
        self.duplicates: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate rows, quarantine bad ones, drop exact duplicates, stop on conflicts."""
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        self.quarantine, self.duplicates = [], []
        seen: Dict[Tuple[str, int], Tuple[Dict[str, Any], int]] = {}
        kept: List[Dict[str, Any]] = []

        for position, original in enumerate(df.to_dict("records"), start=1):
            source_file = original.get("source_file", "")
            source_record = int(original.get("source_record", position))
            reasons, typed = validate_record(original, self.rules)
            if reasons:
                self.quarantine.append({"source_file": source_file, "source_record": source_record,
                                        "reason": "; ".join(reasons),
                                        **{c: original[c] for c in CANONICAL_COLUMNS}})
                continue
            key = (typed["country"], typed["year"])
            if key in seen:
                first, first_record = seen[key]
                if first != typed:
                    raise ConflictingKeyError(
                        f"conflicting key {key}: records {first_record} and {source_record} disagree")
                self.duplicates.append({"source_file": source_file, "source_record": source_record,
                                        "duplicate_of_record": first_record, **typed})
                continue
            seen[key] = (typed, source_record)
            kept.append(typed)

        table = pd.DataFrame(kept, columns=CANONICAL_COLUMNS).astype(
            {"year": "int64", "pop": "int64", "lifeExp": "float64", "gdpPercap": "float64"})
        table = table.sort_values(["country", "year"]).reset_index(drop=True)
        self.counts = {"raw": len(df), "rejected": len(self.quarantine),
                       "duplicates": len(self.duplicates), "canonical": len(table)}
        return table

    def quarantine_frame(self) -> pd.DataFrame:
        """Return quarantined rows with their original values."""
        return pd.DataFrame(self.quarantine, columns=["source_file", "source_record", "reason"] + CANONICAL_COLUMNS)

    def duplicate_frame(self) -> pd.DataFrame:
        """Return the log of removed duplicate copies."""
        return pd.DataFrame(self.duplicates, columns=["source_file", "source_record", "duplicate_of_record"] + CANONICAL_COLUMNS)

    def reconciles(self) -> bool:
        """Check raw = rejected + duplicates + canonical."""
        c = self.counts
        return c["raw"] == c["rejected"] + c["duplicates"] + c["canonical"]
