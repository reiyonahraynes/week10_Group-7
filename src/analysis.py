from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

CANONICAL_COLUMNS: List[str] = ["country", "continent", "year", "lifeExp", "pop", "gdpPercap"]
DEFAULT_COUNTRIES: List[str] = ["Philippines", "Indonesia", "Thailand"]
DEFAULT_MATRIX_YEARS: List[int] = [1982, 1987, 1992, 1997, 2002, 2007]
DEFAULT_BASELINE_YEAR: int = 1982
FLOAT_ABS_TOL: float = 1e-8


def _require_columns(df: pd.DataFrame, required: Set[str]) -> None:
    """Raise ValueError if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Canonical table is missing columns: {sorted(missing)}")


def _as_number(value: Any) -> Any:
    """Convert NumPy scalars to Python numbers."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def asia_selected_year(df: pd.DataFrame, year: int, continent: str = "Asia") -> pd.DataFrame:
    """Add population in millions and life-expectancy rank."""
    _require_columns(df, set(CANONICAL_COLUMNS))
    selected = df.loc[(df["continent"] == continent) & (df["year"] == year)].copy()
    if selected.empty:
        raise ValueError(f"No {continent} records found for year {year}")
    selected["pop_millions"] = selected["pop"] / 1_000_000
    selected["lifeExp_rank"] = (
        selected["lifeExp"].rank(method="min", ascending=False).astype(int)
    )
    return selected.sort_values(["lifeExp_rank", "country"]).reset_index(drop=True)


def country_history(df: pd.DataFrame, country: str = "Philippines") -> pd.DataFrame:
    """Return a country's history with population in millions and life-expectancy change."""
    _require_columns(df, set(CANONICAL_COLUMNS))
    history = df.loc[df["country"] == country].sort_values("year").copy()
    if history.empty:
        raise ValueError(f"No records found for country {country!r}")
    history["pop_millions"] = history["pop"] / 1_000_000
    history["lifeExp_change"] = history["lifeExp"] - history["lifeExp"].iloc[0]
    return history.reset_index(drop=True)


def build_population_matrix(
    df: pd.DataFrame,
    countries: Sequence[str] = tuple(DEFAULT_COUNTRIES),
    years: Sequence[int] = tuple(DEFAULT_MATRIX_YEARS),
) -> np.ndarray:
    """Countries x years population matrix."""
    _require_columns(df, {"country", "year", "pop"})
    country_arr = df["country"].to_numpy()
    year_arr = df["year"].to_numpy()
    pop_arr = df["pop"].to_numpy(dtype=np.int64)

    matrix = np.zeros((len(countries), len(years)), dtype=np.int64)
    for i, country in enumerate(countries):
        for j, year in enumerate(years):
            mask = (country_arr == country) & (year_arr == year)
            found = int(mask.sum())
            if found != 1:
                raise ValueError(
                    f"Expected exactly one record for {country} in {year}, found {found}"
                )
            matrix[i, j] = pop_arr[mask][0]
    return matrix


def broadcast_population_index(
    matrix: np.ndarray, baseline_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Index each row to its baseline column (= 100)."""
    baseline = matrix[:, [baseline_col]]
    index = matrix / baseline * 100.0
    return baseline, index


def population_index_loop(matrix: np.ndarray, baseline_col: int = 0) -> np.ndarray:
    """Loop version of the index, to check broadcasting."""
    rows, cols = matrix.shape
    result = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        base = float(matrix[i, baseline_col])
        for j in range(cols):
            result[i, j] = float(matrix[i, j]) / base * 100.0
    return result


def shape_report(
    matrix: np.ndarray, baseline: np.ndarray, index: np.ndarray
) -> Dict[str, Tuple[int, ...]]:
    """Report shapes before and after broadcasting."""
    return {
        "population_matrix": tuple(matrix.shape),
        "baseline_before_broadcast": tuple(baseline.shape),
        "index_after_broadcast": tuple(index.shape),
    }


def population_index_frame(
    labels: Sequence[str], years: Sequence[int], index: np.ndarray
) -> pd.DataFrame:
    """Return the index as a table with named year columns."""
    frame = pd.DataFrame(index, columns=[str(y) for y in years])
    frame.insert(0, "country", list(labels))
    return frame


def boolean_selection_demo(
    matrix: np.ndarray, labels: Sequence[str], threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Demo Boolean selection and aggregation."""
    last_year = matrix[:, -1]
    cutoff = float(np.median(last_year)) if threshold is None else float(threshold)
    mask = last_year > cutoff
    return {
        "threshold": cutoff,
        "mask": mask,
        "selected_countries": [labels[i] for i in np.flatnonzero(mask)],
        "selected_total": int(matrix[mask, -1].sum()),
        "column_totals": matrix.sum(axis=0),
    }


def asian_population_total_numpy(df: pd.DataFrame, year: int, continent: str = "Asia") -> int:
    """Sum a continent-year population using NumPy only."""
    country_arr = df["continent"].to_numpy()
    year_arr = df["year"].to_numpy()
    pop_arr = df["pop"].to_numpy(dtype=np.int64)
    mask = (country_arr == continent) & (year_arr == year)
    return int(pop_arr[mask].sum())


def country_population_numpy(df: pd.DataFrame, country: str, year: int) -> int:
    """Look up one country-year population with NumPy."""
    mask = (df["country"].to_numpy() == country) & (df["year"].to_numpy() == year)
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected exactly one record for {country} in {year}")
    return int(df["pop"].to_numpy(dtype=np.int64)[mask][0])


def make_check(name: str, expected: Any, actual: Any, tolerance: float = 0.0) -> Dict[str, Any]:
    """Return one validation row."""
    expected, actual = _as_number(expected), _as_number(actual)
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "pass": bool(abs(actual - expected) <= tolerance),
    }


def build_analysis_checks(
    df: pd.DataFrame,
    year: int,
    asia_df: pd.DataFrame,
    history: pd.DataFrame,
    matrix: np.ndarray,
    index: np.ndarray,
    loop_index: np.ndarray,
    baseline_col: int = 0,
    expected_controls: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Analytical and NumPy check rows."""
    controls = expected_controls or {}
    asia_total_np = asian_population_total_numpy(df, year)
    ph_pop_np = country_population_numpy(df, "Philippines", year)
    ph_row = asia_df.loc[asia_df["country"] == "Philippines", "pop"]

    checks = [
        make_check(f"asia_row_count_{year}", int(((df["continent"] == "Asia") & (df["year"] == year)).sum()), len(asia_df)),
        make_check(f"asia_population_total_{year}_pandas_vs_numpy", asia_total_np, int(asia_df["pop"].sum())),
        make_check(f"asia_pop_millions_total_{year}", asia_total_np / 1_000_000, float(asia_df["pop_millions"].sum()), FLOAT_ABS_TOL),
        make_check(f"philippines_population_{year}", ph_pop_np, int(ph_row.iloc[0]) if len(ph_row) else -1),
        make_check("asia_rank_starts_at_1", 1, int(asia_df["lifeExp_rank"].min())),
        make_check("asia_rank_sorted", 1, int(asia_df["lifeExp_rank"].is_monotonic_increasing)),
        make_check("country_history_rows", int((df["country"] == "Philippines").sum()), len(history)),
        make_check("country_history_first_change_zero", 0.0, float(history["lifeExp_change"].iloc[0]), FLOAT_ABS_TOL),
        make_check(
            "country_history_total_change",
            float(history["lifeExp"].iloc[-1] - history["lifeExp"].iloc[0]),
            float(history["lifeExp_change"].iloc[-1]),
            FLOAT_ABS_TOL,
        ),
        make_check("numpy_matrix_rows", len(DEFAULT_COUNTRIES), matrix.shape[0]),
        make_check("numpy_matrix_cols", len(DEFAULT_MATRIX_YEARS), matrix.shape[1]),
        make_check("numpy_index_baseline_is_100", 0.0, float(np.abs(index[:, baseline_col] - 100.0).max()), FLOAT_ABS_TOL),
        make_check("numpy_broadcast_vs_loop", 0.0, float(np.abs(index - loop_index).max()), FLOAT_ABS_TOL),
    ]


    if "asia_population_total" in controls:
        checks.append(make_check(f"control_asia_population_total_{year}", controls["asia_population_total"], asia_total_np))
    if "philippines_population" in controls:
        checks.append(make_check(f"control_philippines_population_{year}", controls["philippines_population"], ph_pop_np))
    if "asia_rows" in controls:
        checks.append(make_check(f"control_asia_rows_{year}", controls["asia_rows"], len(asia_df)))
    if "philippines_rows" in controls:
        checks.append(make_check("control_philippines_rows", controls["philippines_rows"], len(history)))
    return checks


def controls_for_year(manifest_controls: Dict[str, Any], year: int) -> Dict[str, Any]:
    """Map manifest controls to expected_controls for one year"""
    mapping = {
        f"asia_{year}_rows": "asia_rows",
        f"asia_{year}_population": "asia_population_total",
        f"philippines_{year}_population": "philippines_population",
        "philippines_rows": "philippines_rows",
    }
    return {mine: manifest_controls[theirs] for theirs, mine in mapping.items() if theirs in manifest_controls}


def run_analysis(
    df: pd.DataFrame,
    year: int,
    countries: Optional[Sequence[str]] = None,
    matrix_years: Optional[Sequence[int]] = None,
    baseline_year: int = DEFAULT_BASELINE_YEAR,
    expected_controls: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all analysis steps and return the results."""
    countries = list(countries) if countries else list(DEFAULT_COUNTRIES)
    matrix_years = list(matrix_years) if matrix_years else list(DEFAULT_MATRIX_YEARS)
    baseline_col = matrix_years.index(baseline_year)
    audit: List[Dict[str, Any]] = []

    asia_df = asia_selected_year(df, year)
    audit.append({"step": "analysis", "source_file": "canonical_table", "operation": "select_asia_year",
                  "rule": f".loc continent=Asia & year={year}; rank method=min", "rows_before": len(df), "rows_after": len(asia_df)})

    history = country_history(df, "Philippines")
    audit.append({"step": "analysis", "source_file": "canonical_table", "operation": "country_history",
                  "rule": "country=Philippines sorted by year", "rows_before": len(df), "rows_after": len(history)})

    matrix = build_population_matrix(df, countries, matrix_years)
    baseline, index = broadcast_population_index(matrix, baseline_col)
    loop_index = population_index_loop(matrix, baseline_col)
    audit.append({"step": "analysis", "source_file": "canonical_table", "operation": "population_matrix",
                  "rule": f"{len(countries)}x{len(matrix_years)} matrix, baseline {baseline_year}=100",
                  "rows_before": len(df), "rows_after": matrix.shape[0]})

     checks = build_analysis_checks(df, year, asia_df, history, matrix, index, loop_index, baseline_col, expected_controls,
                                   expected_shape=(len(countries), len(matrix_years)))
    return {
        "asia_selected_year": asia_df,
        "country_history": history,
        "population_index": population_index_frame(countries, matrix_years, index),
        "matrix": matrix,
        "baseline": baseline,
        "index": index,
        "loop_index": loop_index,
        "shapes": shape_report(matrix, baseline, index),
        "boolean_demo": boolean_selection_demo(matrix, countries),
        "checks": checks,
        "audit_records": audit,
    }


def write_analysis_outputs(results: Dict[str, Any], out_dir: "str | Path") -> List[Path]:
    """Write the three analysis CSVs to out_dir."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in [
        ("asia_selected_year", "asia_selected_year.csv"),
        ("country_history", "country_history.csv"),
        ("population_index", "population_index.csv"),
    ]:
        path = folder / name
        results[key].to_csv(path, index=False)
        written.append(path)
    return written
