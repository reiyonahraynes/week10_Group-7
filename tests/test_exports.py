import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import exports
from src.compare import apply_schema


def table() -> pd.DataFrame:
    return apply_schema(pd.DataFrame({
        "country": ["Côte d'Ivoire", "Hong Kong, China", "Namibia"],
        "continent": ["Africa", "Asia", "Africa"], "year": [1952, 1957, 2007],
        "lifeExp": [40.477, 61.96, 52.906], "pop": [2977019, 2736300, 2055080],
        "gdpPercap": [1388.594732, 3629.076457, 4811.060429]}))


class ExportTests(unittest.TestCase):
    def test_all_four_formats_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            checks, info = exports.export_and_verify(table(), Path(d))
            self.assertTrue(all(c["pass"] for c in checks), [c for c in checks if not c["pass"]])
            self.assertEqual({i["format"] for i in info}, {"csv", "json", "parquet", "xlsx"})
            for i in info:
                self.assertTrue((Path(d) / i["file"]).is_file())
                self.assertEqual(len(i["sha256"]), 64)

    def test_text_with_comma_and_accent_survives(self):
        with tempfile.TemporaryDirectory() as d:
            exports.export_and_verify(table(), Path(d))
            for fmt in ("csv", "json", "parquet", "xlsx"):
                name, _, reader = exports.EXPORTERS[fmt]
                back = apply_schema(reader(Path(d) / name))
                self.assertIn("Hong Kong, China", set(back["country"]))
                self.assertIn("Côte d'Ivoire", set(back["country"]))

    def test_namibia_is_not_read_as_missing(self):
        with tempfile.TemporaryDirectory() as d:
            exports.export_and_verify(table(), Path(d))
            back = exports.reload_csv(Path(d) / "canonical_dataset.csv")
            self.assertEqual(back["country"].isna().sum(), 0)

    def test_corrupted_export_is_a_failed_check_not_an_exception(self):
        def shifted_reader(path):
            return exports.reload_csv(path).assign(pop=lambda x: x["pop"] + 1)
        patched = dict(exports.EXPORTERS)
        patched["csv"] = ("canonical_dataset.csv", exports.export_csv, shifted_reader)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(exports.EXPORTERS, patched):
            checks, _ = exports.export_and_verify(table(), Path(d))
            self.assertIn("export_csv_values", {c["check"] for c in checks if not c["pass"]})

    def test_unreadable_export_is_reported(self):
        def boom(path):
            raise OSError("cannot read")
        patched = dict(exports.EXPORTERS)
        patched["json"] = ("canonical_dataset.json", exports.export_json, boom)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(exports.EXPORTERS, patched):
            checks, info = exports.export_and_verify(table(), Path(d))
            self.assertIn("export_json_reload", {c["check"] for c in checks if not c["pass"]})
            self.assertNotIn("json", {i["format"] for i in info})


if __name__ == "__main__":
    unittest.main()
