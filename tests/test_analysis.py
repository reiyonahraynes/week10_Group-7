import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    CANONICAL_COLUMNS,
    asia_selected_year,
    asian_population_total_numpy,
    boolean_selection_demo,
    broadcast_population_index,
    build_analysis_checks,
    build_population_matrix,
    controls_for_year,
    country_history,
    country_population_numpy,
    population_index_frame,
    population_index_loop,
    run_analysis,
    write_analysis_outputs,
)

YEARS = [1982, 1987, 1992, 1997, 2002, 2007]
BASE = {
    "Japan": ("Asia", 120_000_000, 70.0),
    "Thailand": ("Asia", 45_000_000, 60.0),
    "Philippines": ("Asia", 50_000_000, 55.0),
    "Indonesia": ("Asia", 150_000_000, 50.0),
    "Cambodia": ("Asia", 7_000_000, 45.0),
    "Nepal": ("Asia", 15_000_000, 45.0),
    "Kenya": ("Africa", 17_000_000, 90.0),
}


def make_fixture() -> pd.DataFrame:
    """Small in-memory canonical table."""
    rows = []
    for country, (continent, pop0, life0) in BASE.items():
        for k, year in enumerate(YEARS):
            rows.append([country, continent, year, life0 + 2.0 * k, pop0 + 1_000_000 * k, 1000.0 + 10 * k])
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


class AsiaSelectedYearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_fixture()

    def test_filters_continent_and_year(self) -> None:
        out = asia_selected_year(self.df, 2007)
        self.assertEqual(len(out), 6)
        self.assertNotIn("Kenya", set(out["country"]))
        self.assertEqual(set(out["year"]), {2007})

    def test_rank_uses_method_min_and_descending(self) -> None:
        out = asia_selected_year(self.df, 2007).set_index("country")
        self.assertEqual(out.loc["Japan", "lifeExp_rank"], 1)
        self.assertEqual(out.loc["Cambodia", "lifeExp_rank"], 5)
        self.assertEqual(out.loc["Nepal", "lifeExp_rank"], 5)

    def test_sorted_by_rank_then_country(self) -> None:
        out = asia_selected_year(self.df, 2007)
        self.assertEqual(list(out["country"]), ["Japan", "Thailand", "Philippines", "Indonesia", "Cambodia", "Nepal"])

    def test_population_in_millions(self) -> None:
        out = asia_selected_year(self.df, 2007).set_index("country")
        self.assertAlmostEqual(out.loc["Philippines", "pop_millions"], 55.0, places=8)

    def test_changing_year_regenerates_table(self) -> None:
        out = asia_selected_year(self.df, 1997).set_index("country")
        self.assertEqual(set(out["year"]), {1997})
        self.assertAlmostEqual(out.loc["Philippines", "pop_millions"], 53.0, places=8)

    def test_unknown_year_raises(self) -> None:
        with self.assertRaises(ValueError):
            asia_selected_year(self.df, 1800)

    def test_missing_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            asia_selected_year(self.df.drop(columns=["lifeExp"]), 2007)


class CountryHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_fixture()

    def test_rows_sorted_by_year(self) -> None:
        out = country_history(self.df, "Philippines")
        self.assertEqual(list(out["year"]), YEARS)

    def test_life_expectancy_change_from_first_year(self) -> None:
        out = country_history(self.df, "Philippines")
        self.assertEqual(out["lifeExp_change"].iloc[0], 0.0)
        self.assertAlmostEqual(out["lifeExp_change"].iloc[-1], 10.0, places=8)

    def test_population_in_millions(self) -> None:
        out = country_history(self.df, "Philippines")
        self.assertAlmostEqual(out["pop_millions"].iloc[0], 50.0, places=8)

    def test_unknown_country_raises(self) -> None:
        with self.assertRaises(ValueError):
            country_history(self.df, "Atlantis")


class PopulationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_fixture()
        self.labels = ["Philippines", "Indonesia", "Thailand"]
        self.matrix = build_population_matrix(self.df, self.labels, YEARS)

    def test_shape_and_values(self) -> None:
        self.assertEqual(self.matrix.shape, (3, 6))
        self.assertEqual(int(self.matrix[0, 0]), 50_000_000)
        self.assertEqual(int(self.matrix[1, -1]), 155_000_000)

    def test_missing_country_year_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_population_matrix(self.df, ["Philippines", "Atlantis"], YEARS)

    def test_broadcast_shapes(self) -> None:
        baseline, index = broadcast_population_index(self.matrix, 0)
        self.assertEqual(baseline.shape, (3, 1))
        self.assertEqual(index.shape, (3, 6))

    def test_baseline_column_is_100(self) -> None:
        _, index = broadcast_population_index(self.matrix, 0)
        np.testing.assert_allclose(index[:, 0], 100.0, rtol=0, atol=1e-8)

    def test_broadcast_matches_independent_loop(self) -> None:
        _, index = broadcast_population_index(self.matrix, 0)
        loop = population_index_loop(self.matrix, 0)
        np.testing.assert_allclose(index, loop, rtol=0, atol=1e-8)

    def test_known_index_value(self) -> None:
        _, index = broadcast_population_index(self.matrix, 0)
        self.assertAlmostEqual(index[0, -1], 110.0, places=8)

    def test_index_frame_layout(self) -> None:
        _, index = broadcast_population_index(self.matrix, 0)
        frame = population_index_frame(self.labels, YEARS, index)
        self.assertEqual(list(frame.columns), ["country"] + [str(y) for y in YEARS])
        self.assertEqual(list(frame["country"]), self.labels)

    def test_boolean_selection_and_aggregation(self) -> None:
        demo = boolean_selection_demo(self.matrix, self.labels, threshold=100_000_000)
        self.assertEqual(demo["selected_countries"], ["Indonesia"])
        self.assertEqual(demo["selected_total"], 155_000_000)
        self.assertEqual(demo["column_totals"].shape, (6,))


class IndependentTotalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_fixture()

    def test_numpy_total_matches_hand_calculation(self) -> None:
        self.assertEqual(asian_population_total_numpy(self.df, 2007), 417_000_000)

    def test_numpy_total_matches_pandas_summary(self) -> None:
        pandas_total = int(asia_selected_year(self.df, 2007)["pop"].sum())
        self.assertEqual(asian_population_total_numpy(self.df, 2007), pandas_total)

    def test_total_changes_with_year(self) -> None:
        self.assertEqual(asian_population_total_numpy(self.df, 1997), 405_000_000)

    def test_country_population_lookup(self) -> None:
        self.assertEqual(country_population_numpy(self.df, "Philippines", 2007), 55_000_000)


class CalculationCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_fixture()
        self.controls = {"asia_population_total": 417_000_000, "philippines_population": 55_000_000, "asia_rows": 6}

    def test_all_checks_pass_on_clean_data(self) -> None:
        results = run_analysis(self.df, 2007, expected_controls=self.controls)
        failed = [c for c in results["checks"] if not c["pass"]]
        self.assertEqual(failed, [])
        self.assertTrue(any(c["check"].startswith("control_") for c in results["checks"]))

    def test_check_rows_have_required_keys(self) -> None:
        results = run_analysis(self.df, 2007)
        for row in results["checks"]:
            self.assertEqual(set(row), {"check", "expected", "actual", "tolerance", "pass"})

    def test_wrong_control_value_fails(self) -> None:
        bad = dict(self.controls, asia_population_total=417_000_001)
        results = run_analysis(self.df, 2007, expected_controls=bad)
        self.assertFalse(all(c["pass"] for c in results["checks"]))

    def test_tampered_summary_is_detected(self) -> None:
        results = run_analysis(self.df, 2007)
        tampered = results["asia_selected_year"].copy()
        tampered.loc[0, "pop"] += 1
        checks = build_analysis_checks(
            self.df, 2007, tampered, results["country_history"],
            results["matrix"], results["index"], results["loop_index"],
        )
        self.assertFalse(all(c["pass"] for c in checks))

    def test_audit_records_have_required_keys(self) -> None:
        results = run_analysis(self.df, 2007)
        keys = {"step", "source_file", "operation", "rule", "rows_before", "rows_after"}
        for record in results["audit_records"]:
            self.assertEqual(set(record), keys)

    def test_outputs_written_to_staging_folder(self) -> None:
        results = run_analysis(self.df, 2007)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_analysis_outputs(results, Path(tmp) / "staging")
            self.assertEqual(sorted(p.name for p in paths),
                             ["asia_selected_year.csv", "country_history.csv", "population_index.csv"])
            reloaded = pd.read_csv(Path(tmp) / "staging" / "population_index.csv")
            self.assertEqual(reloaded.shape, (3, 7))


class ManifestControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_controls = {
            "asia_2007_rows": 6,
            "asia_2007_population": 417_000_000,
            "philippines_2007_population": 55_000_000,
            "philippines_rows": 6,
        }

    def test_controls_translated_for_2007(self) -> None:
        got = controls_for_year(self.manifest_controls, 2007)
        self.assertEqual(got, {"asia_rows": 6, "asia_population_total": 417_000_000,
                               "philippines_population": 55_000_000, "philippines_rows": 6})

    def test_year_specific_controls_dropped_for_other_years(self) -> None:
        got = controls_for_year(self.manifest_controls, 1997)
        self.assertEqual(got, {"philippines_rows": 6})

    def test_changed_year_does_not_fail_on_2007_controls(self) -> None:
        results = run_analysis(make_fixture(), 1997, expected_controls=controls_for_year(self.manifest_controls, 1997))
        self.assertTrue(all(c["pass"] for c in results["checks"]))

    def test_philippines_row_control_checked(self) -> None:
        results = run_analysis(make_fixture(), 2007, expected_controls=controls_for_year(self.manifest_controls, 2007))
        names = [c["check"] for c in results["checks"]]
        self.assertIn("control_philippines_rows", names)
        self.assertTrue(all(c["pass"] for c in results["checks"]))


if __name__ == "__main__":
    unittest.main()