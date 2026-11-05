import unittest

import pandas as pd

from src.compare import (ComparisonError, SchemaError, apply_schema, compare_all,
                         compare_tables, schema_dtypes)


def sample() -> pd.DataFrame:
    return pd.DataFrame({"country": ["B", "A"], "continent": ["Asia", "Asia"], "year": [1952, 1957],
                         "lifeExp": [50.5, 60.25], "pop": [1000, 2000], "gdpPercap": [100.5, 200.5]})


class SchemaTests(unittest.TestCase):
    def test_applies_dtypes_order_and_sorts(self):
        text = sample().astype(str)
        out = apply_schema(text[["pop", "country", "continent", "year", "lifeExp", "gdpPercap"]])
        self.assertEqual(list(out["country"]), ["A", "B"])
        self.assertEqual(schema_dtypes(out), {"country": "text", "continent": "text", "year": "int64",
                                              "lifeExp": "float64", "pop": "int64", "gdpPercap": "float64"})

    def test_missing_column_raises(self):
        with self.assertRaises(SchemaError):
            apply_schema(sample().drop(columns=["pop"]))

    def test_fractional_integer_column_raises(self):
        bad = sample()
        bad["pop"] = [1000.5, 2000.0]
        with self.assertRaises(SchemaError):
            apply_schema(bad)

    def test_non_numeric_raises(self):
        bad = sample().astype(object)
        bad.loc[0, "lifeExp"] = "abc"
        with self.assertRaises(SchemaError):
            apply_schema(bad)


class CompareTests(unittest.TestCase):
    def test_same_data_different_row_order_agrees(self):
        a = apply_schema(sample())
        b = apply_schema(sample().iloc[::-1])
        self.assertTrue(compare_tables(a, b)["agree"])

    def test_value_mismatch_is_reported_by_key(self):
        a = apply_schema(sample())
        b = a.copy()
        b.loc[b["country"] == "A", "pop"] = 999
        rep = compare_tables(a, b, "x", "y")
        self.assertFalse(rep["agree"])
        self.assertEqual(rep["mismatches"][0]["country"], "A")
        self.assertEqual(rep["mismatches"][0]["column"], "pop")

    def test_missing_and_extra_rows(self):
        a = apply_schema(sample())
        b = a.iloc[:1].copy()
        rep = compare_tables(a, b)
        self.assertEqual(len(rep["only_left"]), 1)
        self.assertFalse(rep["agree"])

    def test_float_within_tolerance_agrees_beyond_does_not(self):
        a = apply_schema(sample())
        b = a.copy()
        b["gdpPercap"] += 1e-12
        self.assertTrue(compare_tables(a, b)["agree"])
        b["gdpPercap"] += 1e-3
        self.assertFalse(compare_tables(a, b)["agree"])

    def test_duplicate_key_raises(self):
        a = apply_schema(sample())
        dup = pd.concat([a, a.iloc[:1]], ignore_index=True)
        with self.assertRaises(ComparisonError):
            compare_tables(a, dup)

    def test_compare_all_needs_two_tables(self):
        with self.assertRaises(ComparisonError):
            compare_all({"only": apply_schema(sample())})

    def test_compare_all_summary(self):
        a = apply_schema(sample())
        b = a.copy()
        b.loc[0, "lifeExp"] = 1.0
        summary = compare_all({"a": a, "b": b, "c": a.copy()})
        self.assertEqual(list(summary["agree"]), [False, True])


if __name__ == "__main__":
    unittest.main()
