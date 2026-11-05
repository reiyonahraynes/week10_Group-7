
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.analysis import controls_for_year, run_analysis, write_analysis_outputs
from src.compare import CANONICAL_ORDER, compare_all
from src.exports import export_and_verify
from src.readers import (FileFormatError, load_configuration, read_csv_records,
                         read_excel_records, validate_input_file)
from src.readers_json_parquet import map_canonical_columns, read_json_records, read_parquet_records
from src.validation import ConflictingKeyError, ValidationRun, build_rules

CATEGORIES = (
    "config_missing", "manifest_missing", "input_missing", "unsupported_format",
    "integrity_mismatch", "malformed_file", "missing_column", "conflicting_key",
    "count_mismatch", "format_disagreement", "analysis_failure", "export_failure",
)
COUNT_KEYS = (("raw_rows", "raw"), ("rejected_rows", "rejected"),
              ("exact_duplicates_removed", "duplicates"), ("canonical_rows", "canonical"))


class ProcessingFailure(Exception):
    """A run failed. category is one of CATEGORIES; nothing was published."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"[{category}] {message}")
        self.category, self.message = category, message


def classify_error(exc: Exception) -> ProcessingFailure:
    """Map an error from the other modules to a ProcessingFailure category."""
    if isinstance(exc, ProcessingFailure):
        return exc
    text, cat = str(exc), "malformed_file"
    if isinstance(exc, ConflictingKeyError):
        cat = "conflicting_key"
    elif isinstance(exc, FileNotFoundError):
        cat = ("config_missing" if "Configuration" in text
               else "manifest_missing" if "Manifest" in text else "input_missing")
    elif isinstance(exc, FileFormatError):
        if "Unsupported file format" in text:
            cat = "unsupported_format"
        elif "mismatch" in text and ("SHA-256" in text or "Byte size" in text):
            cat = "integrity_mismatch"
        elif "Missing required canonical column" in text:
            cat = "missing_column"
    elif isinstance(exc, ValueError) and "missing required columns" in text:
        cat = "missing_column"
    else:
        text = f"{type(exc).__name__}: {text}"
    return ProcessingFailure(cat, text)


class ProcessingRun:
    """One end-to-end run: run() for the four formats, run_single_file(path) for one file."""

    def __init__(self, config_path: "str | Path" = "config.json") -> None:
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent
        self.staging_dir, self.output_dir = self.base_dir / "staging", self.base_dir / "outputs"
        self.audit: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.published = False
        self.year_override: Optional[int] = None
    def log(self, step: str, source_file: str, operation: str, rule: str,
            rows_before: Optional[int], rows_after: Optional[int]) -> None:
        """Append one audit-log row."""
        self.audit.append({"seq": len(self.audit) + 1, "step": step, "source_file": source_file,
                           "operation": operation, "rule": rule,
                           "rows_before": rows_before, "rows_after": rows_after})

    def check(self, group: str, name: str, expected: Any, actual: Any, detail: str = "") -> None:
        """Record one check; group is the failure category used if it fails."""
        self.checks.append({"group": group, "check": name, "expected": expected, "actual": actual,
                            "tolerance": 0.0, "pass": bool(expected == actual), "detail": detail})

    def _add_rows(self, group: str, rows: List[Dict[str, Any]]) -> None:
        self.checks.extend({"group": group, "detail": "", **r, "pass": bool(r["pass"])} for r in rows)

    def _fail_if_any(self, *groups: str) -> None:
        """Raise for the first failed check in the given groups."""
        for r in self.checks:
            if r["group"] in groups and not r["pass"]:
                raise ProcessingFailure(r["group"], f"check '{r['check']}' failed: expected "
                                        f"{r['expected']}, got {r['actual']} {r['detail']}".strip())
    def load(self) -> None:
        """Load config and manifest; folders resolve relative to the config file."""
        previous = Path.cwd()
        os.chdir(self.base_dir)
        try:
            self.config = load_configuration(self.config_path.name)
        finally:
            os.chdir(previous)
        self.manifest = self.config["manifest"]
        self.staging_dir = self.base_dir / self.config.get("staging_dir", "staging")
        self.output_dir = self.base_dir / self.config.get("output_dir", "outputs")
        self.data_dir = self.base_dir / self.config.get("data_dir", "data")

    def _read_spec(self, spec: Dict[str, Any]) -> pd.DataFrame:
        """Read one manifest entry, check size and hash, rename to canonical columns."""
        path, sha, size = self.data_dir / spec["file"], spec.get("sha256"), spec.get("bytes")
        readers: Dict[str, Callable[[], pd.DataFrame]] = {
            "csv": lambda: read_csv_records(path, spec["encoding"], spec["separator"],
                                            spec.get("comment_prefix"), spec.get("ignore_columns"), sha, size),
            "xlsx": lambda: read_excel_records(path, spec["sheet_name"], spec.get("engine", "openpyxl"), sha, size),
            "json": lambda: read_json_records(path, spec["record_path"], sha, size),
            "parquet": lambda: read_parquet_records(path, sha, size),
        }
        if spec["format"] not in readers:
            raise FileFormatError(f"Unsupported file format '{spec['format']}' for '{spec['file']}'.")
        return map_canonical_columns(readers[spec["format"]](), spec["source_to_canonical"], list(CANONICAL_ORDER))

    def _read_single(self, path: Path) -> pd.DataFrame:
        """Read one file that already uses canonical column names."""
        validate_input_file(path)
        s = self.config.get("failure_test_csv_settings", {})
        readers: Dict[str, Callable[[], pd.DataFrame]] = {
            ".csv": lambda: read_csv_records(path, s.get("encoding", "utf-8"), s.get("separator", ","),
                                             s.get("comment_prefix"), s.get("ignore_columns")),
            ".xlsx": lambda: read_excel_records(path),
            ".json": lambda: read_json_records(path),
        }
        df = readers.get(path.suffix.lower(), lambda: read_parquet_records(path))()
        return map_canonical_columns(df, {}, list(CANONICAL_ORDER))

    def _process_source(self, name: str, df: pd.DataFrame, expected: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Validate one recovered table; record audit rows and count checks."""
        validator = ValidationRun(build_rules(self.manifest))
        table, c = validator.process(df), validator.counts
        self.log("read", name, "read_and_map", "read as text, rename to canonical columns", len(df), len(df))
        self.log("validate", name, "quarantine_invalid", "rules in manifest", c["raw"], c["raw"] - c["rejected"])
        self.log("deduplicate", name, "remove_exact_duplicates", "same (country, year), identical values",
                 c["raw"] - c["rejected"], c["canonical"])
        self.check("count_mismatch", f"{name}_reconciles", True, validator.reconciles(),
                   f"raw {c['raw']} = rejected {c['rejected']} + duplicates {c['duplicates']} + canonical {c['canonical']}")
        for key, mine in COUNT_KEYS:
            if expected and key in expected:
                self.check("count_mismatch", f"{name}_{key}", expected[key], c[mine])
        return {"table": table, "quarantine": validator.quarantine_frame(),
                "duplicates": validator.duplicate_frame(), "counts": c}
    def run(self) -> Dict[str, Any]:
        """Process the four-format dataset and publish the results."""
        expected = lambda: self.manifest.get("expected_per_file")
        return self._execute(lambda: {s["file"]: self._process_source(s["file"], self._read_spec(s), expected())
                                      for s in self.manifest["files"]})

    def run_single_file(self, path: "str | Path") -> Dict[str, Any]:
        """Process one canonical-named file (used for the failure-test files)."""
        target = Path(path)
        return self._execute(lambda: {target.name: self._process_source(target.name, self._read_single(target), None)})

    def _execute(self, gather: Callable[[], Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        self.audit, self.checks, self.published = [], [], False
        try:
            try:
                self.load()
            except Exception as exc:
                raise classify_error(exc) from exc
            if self.staging_dir.exists():
                shutil.rmtree(self.staging_dir)
            self.staging_dir.mkdir(parents=True)
            summary = self._run_steps(gather())
            self._publish()
            summary["published_to"] = str(self.output_dir)
            return summary
        except Exception as exc:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            raise classify_error(exc) from exc

    def _run_steps(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        names = list(sources)
        tables = {n: sources[n]["table"] for n in names}
        canonical = tables[names[0]]
        out = self.staging_dir

        comparison = compare_all(tables) if len(tables) > 1 else pd.DataFrame()
        for _, r in comparison.iterrows():
            self.check("format_disagreement", f"{r['compared']}_agrees_with_{r['reference']}", True,
                       bool(r["agree"]), f"{r['only_in_reference']} missing, {r['only_in_compared']} extra, "
                       f"{r['value_mismatches']} value mismatches")
        if len(names) > 1:
            self.log("compare", "all", "key_comparison", "match on (country, year); floats within 1e-9",
                     len(names), len(names))
        controls = self.manifest.get("controls", {})
        if len(canonical) > 100:
            if "countries" in controls:
                self.check("count_mismatch", "control_country_count", controls["countries"],
                           int(canonical["country"].nunique()))
            rows = self.manifest.get("expected_per_file", {}).get("canonical_rows")
            if rows and self.manifest.get("files"):
                self.check("count_mismatch", "canonical_row_count", rows, len(canonical))
        self._fail_if_any("count_mismatch", "format_disagreement")

        year = int(self.year_override if self.year_override is not None else self.config.get("reporting_year", 2007))
        matrix_years = self.config.get("matrix_years")
        baseline = self.config.get("baseline_year") or (
            1982 if not matrix_years or 1982 in matrix_years else matrix_years[0])
        try:
            analysis = run_analysis(canonical, year, self.config.get("matrix_countries"), matrix_years,
                                    int(baseline), controls_for_year(controls, year))
        except Exception as exc:
            raise ProcessingFailure("analysis_failure", f"{type(exc).__name__}: {exc}") from exc
        self._add_rows("analysis_failure", analysis["checks"])
        for rec in analysis["audit_records"]:
            self.log(rec["step"], rec["source_file"], rec["operation"], rec["rule"],
                     rec["rows_before"], rec["rows_after"])
        self._fail_if_any("analysis_failure")

        write_analysis_outputs(analysis, out)
        export_checks, export_info = export_and_verify(canonical, out / "exports")
        self._add_rows("export_failure", export_checks)
        self.log("export", "canonical_table", "export_four_formats",
                 "csv, json, parquet, xlsx; reload and compare", len(canonical), len(canonical))
        self._fail_if_any("export_failure")

        quarantine = pd.concat([sources[n]["quarantine"] for n in names], ignore_index=True)
        duplicates = pd.concat([sources[n]["duplicates"] for n in names], ignore_index=True)
        quarantine.to_csv(out / "quarantine.csv", index=False)
        duplicates.to_csv(out / "duplicates_removed.csv", index=False)
        if not comparison.empty:
            comparison.to_csv(out / "format_comparison.csv", index=False)
        asia = analysis["asia_selected_year"]
        ph = canonical[(canonical["country"] == "Philippines") & (canonical["year"] == year)]
        self.log("publish", "all", "write_reports", "audit log, checks and summary", None, None)
        summary = {
            "status": "success", "reporting_year": year,
            "files": {n: sources[n]["counts"] for n in names},
            "total_quarantined": len(quarantine), "total_duplicates_removed": len(duplicates),
            "canonical_rows": len(canonical), "countries": int(canonical["country"].nunique()),
            f"asia_{year}_rows": len(asia), f"asia_{year}_population": int(asia["pop"].sum()),
            f"philippines_{year}_population": int(ph["pop"].iloc[0]),
            "checks_total": len(self.checks), "checks_passed": sum(bool(r["pass"]) for r in self.checks),
            "exports": export_info,
        }
        pd.DataFrame(self.audit).to_csv(out / "audit_log.csv", index=False)
        pd.DataFrame(self.checks).to_csv(out / "validation_checks.csv", index=False)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def _publish(self) -> None:
        """Swap staging into the output folder by renames, so readers never see a half-written folder."""
        if any(not r["pass"] for r in self.checks):
            raise ProcessingFailure("analysis_failure", "checks failed; not publishing")
        tmp, old = (self.output_dir.with_name(self.output_dir.name + s) for s in (".tmp", ".old"))
        for leftover in (tmp, old):
            shutil.rmtree(leftover, ignore_errors=True)
        shutil.move(str(self.staging_dir), str(tmp))
        if self.output_dir.exists():
            self.output_dir.rename(old)
        tmp.rename(self.output_dir)
        shutil.rmtree(old, ignore_errors=True)
        self.published = True
