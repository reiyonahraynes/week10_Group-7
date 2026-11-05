import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import main as cli
from src import exports
from src.pipeline import CATEGORIES, ProcessingFailure, ProcessingRun
from tests.support import Sandbox


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)

    def run_ok(self):
        return ProcessingRun(self.sb.config_path).run()

    def assert_failure(self, category, runner=None):
        runner = runner or (lambda: ProcessingRun(self.sb.config_path).run())
        with self.assertRaises(ProcessingFailure) as ctx:
            runner()
        self.assertEqual(ctx.exception.category, category, str(ctx.exception))
        self.assertFalse(self.sb.outputs.exists(), "nothing may be published after a failure")
        self.assertFalse(self.sb.staging.exists(), "staging must be cleaned up")


class SuccessfulRunTests(SandboxCase):
    def test_headline_numbers(self):
        s = self.run_ok()
        self.assertEqual(s["canonical_rows"], 1704)
        self.assertEqual(s["total_quarantined"], 24)
        self.assertEqual(s["total_duplicates_removed"], 8)
        self.assertEqual(s["countries"], 142)
        self.assertEqual(s["asia_2007_rows"], 33)
        self.assertEqual(s["asia_2007_population"], 3811953827)
        self.assertEqual(s["philippines_2007_population"], 91077287)
        self.assertEqual(s["checks_passed"], s["checks_total"])

    def test_every_file_reconciles(self):
        for name, c in self.run_ok()["files"].items():
            self.assertEqual(c["raw"], 1712, name)
            self.assertEqual(c["raw"], c["rejected"] + c["duplicates"] + c["canonical"], name)
            self.assertEqual((c["rejected"], c["duplicates"], c["canonical"]), (6, 2, 1704), name)

    def test_published_files_exist_and_staging_is_gone(self):
        self.run_ok()
        expected = ["asia_selected_year.csv", "country_history.csv", "population_index.csv", "quarantine.csv",
                    "duplicates_removed.csv", "audit_log.csv", "validation_checks.csv",
                    "format_comparison.csv", "run_summary.json"]
        for name in expected:
            self.assertTrue((self.sb.outputs / name).is_file(), name)
        for name in ("csv", "json", "parquet", "xlsx"):
            self.assertTrue((self.sb.outputs / "exports" / f"canonical_dataset.{name}").is_file())
        self.assertFalse(self.sb.staging.exists())

    def test_quarantine_and_duplicate_logs(self):
        self.run_ok()
        q = pd.read_csv(self.sb.outputs / "quarantine.csv", keep_default_na=False)
        d = pd.read_csv(self.sb.outputs / "duplicates_removed.csv")
        self.assertEqual(len(q), 24)
        self.assertEqual(len(d), 8)
        self.assertEqual(q.groupby("source_file").size().tolist(), [6, 6, 6, 6])
        self.assertTrue((q["reason"].str.len() > 0).all())

    def test_audit_log_has_a_row_per_stage(self):
        self.run_ok()
        audit = pd.read_csv(self.sb.outputs / "audit_log.csv")
        for step in ("read", "validate", "deduplicate", "compare", "analysis", "export", "publish"):
            self.assertIn(step, set(audit["step"]))
        self.assertEqual(list(audit["seq"]), list(range(1, len(audit) + 1)))

    def test_all_checks_in_file_pass(self):
        self.run_ok()
        checks = pd.read_csv(self.sb.outputs / "validation_checks.csv")
        self.assertTrue(checks["pass"].all())
        self.assertGreater(len(checks), 40)

    def test_run_is_reproducible(self):
        self.run_ok()
        first = (self.sb.outputs / "exports" / "canonical_dataset.csv").read_bytes()
        self.run_ok()
        self.assertEqual(first, (self.sb.outputs / "exports" / "canonical_dataset.csv").read_bytes())

    def test_reporting_year_from_config_regenerates_analysis(self):
        self.sb.edit_config(reporting_year=1997)
        s = self.run_ok()
        self.assertEqual(s["reporting_year"], 1997)
        asia = pd.read_csv(self.sb.outputs / "asia_selected_year.csv")
        self.assertEqual(set(asia["year"]), {1997})
        self.assertEqual(s["asia_1997_population"], int(asia["pop"].sum()))
        self.assertEqual(s["philippines_1997_population"], 75012988)

    def test_year_override_argument(self):
        run = ProcessingRun(self.sb.config_path)
        run.year_override = 1997
        self.assertEqual(run.run()["reporting_year"], 1997)

    def test_configured_matrix_selection(self):
        self.sb.edit_config(matrix_countries=["Japan", "China"], matrix_years=[1992, 2002, 2007],
                            baseline_year=1992)
        self.run_ok()
        index = pd.read_csv(self.sb.outputs / "population_index.csv")
        self.assertEqual(index.shape, (2, 4))
        self.assertTrue((index["1992"] == 100.0).all())
        checks = pd.read_csv(self.sb.outputs / "validation_checks.csv")
        self.assertTrue(checks["pass"].all())

    def test_previous_outputs_replaced_not_mixed(self):
        self.sb.outputs.mkdir()
        (self.sb.outputs / "stale.txt").write_text("old")
        self.run_ok()
        self.assertFalse((self.sb.outputs / "stale.txt").exists())


class FailureCategoryTests(SandboxCase):
    """One test per failure category. Each must publish nothing."""

    def test_01_config_missing(self):
        self.assert_failure("config_missing", lambda: ProcessingRun(self.sb.dir / "absent.json").run())

    def test_02_manifest_missing(self):
        self.sb.edit_config(manifest_path="data/absent_manifest.json")
        self.assert_failure("manifest_missing")

    def test_03_input_missing(self):
        (self.sb.dir / "data" / "country_records.json").unlink()
        self.assert_failure("input_missing")

    def test_03b_single_missing_file(self):
        self.assert_failure("input_missing",
                            lambda: ProcessingRun(self.sb.config_path).run_single_file(self.sb.dir / "data" / "none.csv"))

    def test_04_unsupported_format(self):
        self.assert_failure("unsupported_format", lambda: ProcessingRun(self.sb.config_path)
                            .run_single_file(self.sb.dir / "data" / "unsupported.txt"))

    def test_05_integrity_mismatch_size(self):
        with (self.sb.dir / "data" / "country_records.parquet").open("ab") as f:
            f.write(b"x")
        self.assert_failure("integrity_mismatch")

    def test_05b_integrity_mismatch_hash_same_size(self):
        path = self.sb.dir / "data" / "country_records.json"
        raw = bytearray(path.read_bytes())
        raw[-5] = ord("x") if raw[-5] != ord("x") else ord("y")
        path.write_bytes(bytes(raw))
        self.assert_failure("integrity_mismatch")

    def test_06_malformed_file(self):
        self.sb.drop_integrity("country_records.json")
        (self.sb.dir / "data" / "country_records.json").write_text("{not json", encoding="utf-8")
        self.assert_failure("malformed_file")

    def test_06b_wrong_sheet_name(self):
        self.sb.edit_manifest(lambda m: [s.update(sheet_name="nope") for s in m["files"] if s["format"] == "xlsx"])
        self.assert_failure("malformed_file")

    def test_07_missing_column(self):
        self.assert_failure("missing_column", lambda: ProcessingRun(self.sb.config_path)
                            .run_single_file(self.sb.dir / "data" / "missing_column.csv"))

    def test_07b_missing_column_in_manifest_mapping(self):
        def fn(m):
            for s in m["files"]:
                if s["file"] == "country_records.csv":
                    del s["source_to_canonical"]["PopulationPersons"]
        self.sb.edit_manifest(fn)
        self.assert_failure("missing_column")

    def test_08_conflicting_key(self):
        self.assert_failure("conflicting_key", lambda: ProcessingRun(self.sb.config_path)
                            .run_single_file(self.sb.dir / "data" / "conflicting_key.csv"))

    def test_09_count_mismatch(self):
        self.sb.edit_manifest(lambda m: m["expected_per_file"].update(rejected_rows=7))
        self.assert_failure("count_mismatch")

    def test_10_format_disagreement(self):
        """Change one valid value in the JSON copy so the four formats disagree."""
        path = self.sb.dir / "data" / "country_records.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = [(r["identity"]["country"], r["identity"]["year"]) for r in data["records"]]
        target = next(i for i, k in enumerate(keys) if keys.count(k) == 1 and k[0] == "Japan")
        data["records"][target]["measures"]["lifeExp"] = 1.5
        path.write_text(json.dumps(data), encoding="utf-8")
        self.sb.drop_integrity("country_records.json")
        self.assert_failure("format_disagreement")

    def test_11_analysis_failure_unknown_country(self):
        self.sb.edit_config(matrix_countries=["Atlantis"])
        self.assert_failure("analysis_failure")

    def test_11b_analysis_failure_bad_year(self):
        self.sb.edit_config(reporting_year=1999)
        self.assert_failure("analysis_failure")

    def test_11c_analysis_control_mismatch(self):
        self.sb.edit_manifest(lambda m: m["controls"].update(asia_2007_population=1))
        self.assert_failure("analysis_failure")

    def test_12_export_failure(self):
        def broken_reader(path):
            return exports.reload_csv(path).assign(pop=lambda x: x["pop"] + 1)
        patched = dict(exports.EXPORTERS)
        patched["csv"] = ("canonical_dataset.csv", exports.export_csv, broken_reader)
        with mock.patch.dict(exports.EXPORTERS, patched):
            self.assert_failure("export_failure")

    def test_all_twelve_categories_are_listed(self):
        self.assertEqual(len(CATEGORIES), 12)
        self.assertEqual(len(set(CATEGORIES)), 12)


class FailureKeepsEarlierOutputs(SandboxCase):
    def test_failed_run_does_not_touch_existing_outputs(self):
        self.run_ok()
        before = (self.sb.outputs / "run_summary.json").read_bytes()
        self.sb.edit_config(matrix_countries=["Atlantis"])
        with self.assertRaises(ProcessingFailure):
            self.run_ok()
        self.assertEqual(before, (self.sb.outputs / "run_summary.json").read_bytes())
        self.assertFalse(self.sb.staging.exists())


class CommandLineTests(SandboxCase):
    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--config", str(self.sb.config_path), *args])
        return code, out.getvalue(), err.getvalue()

    def test_success_exit_zero(self):
        code, out, _ = self.call()
        self.assertEqual(code, 0)
        self.assertIn("3,811,953,827", out)
        self.assertIn("91,077,287", out)

    def test_year_flag(self):
        code, out, _ = self.call("--year", "1997")
        self.assertEqual(code, 0)
        self.assertIn("Asia 1997", out)

    def test_failure_files_exit_one_and_publish_nothing(self):
        for name in ("conflicting_key.csv", "missing_column.csv", "unsupported.txt", "does_not_exist.csv"):
            code, _, err = self.call("--input", str(self.sb.dir / "data" / name))
            self.assertEqual(code, 1, name)
            self.assertIn("Nothing was published", err)
            self.assertFalse(self.sb.outputs.exists(), name)

    def test_missing_config_exit_one(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--config", str(self.sb.dir / "absent.json")])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
