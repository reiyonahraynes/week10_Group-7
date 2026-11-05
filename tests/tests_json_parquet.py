import unittest
from pathlib import Path
from src.readers import FileFormatError, load_configuration
from src.member2_readers import read_json_records, read_parquet_records, map_canonical_columns


class TestMember2Readers(unittest.TestCase):

    def setUp(self) -> None:
        self.config = load_configuration("config.json")
        self.data_dir = Path(self.config["data_dir"])
        self.json_spec = self.config["file_specs"]["country_records.json"]
        self.parquet_spec = self.config["file_specs"]["country_records.parquet"]
        self.canonical_cols = self.config["manifest"]["canonical_columns"]

    def test_json_reader(self) -> None:
        json_path = self.data_dir / self.json_spec["file"]
        df = read_json_records(
            file_path = json_path,
            records_key = self.json_spec.get("records_key", "records"),
            expected_sha256 = self.json_spec["sha256"],
            expected_bytes = self.json_spec["bytes"],
        )
        self.assertEqual(len(df), 1712)
        self.assertEqual(int(df["source_record"].iloc[0]), 1)
        self.assertEqual(int(df["source_record"].iloc[-1]), 1712)
        self.assertEqual(df["source_file"].iloc[0], self.json_spec["file"])
    
    def test_parquet_reader(self) -> None:
        parquet_path = self.data_dir / self.parquet_spec["file"]
        df = read_parquet_records(
            file_path = parquet_path,
            expected_sha256 = self.parquet_spec["sha256"],
            expected_bytes = self.parquet_spec["bytes"],
        )
        self.assertEqual(len(df), 1712)
        self.assertEqual(int(df["source_record"].iloc[0]), 1)
        self.assertEqual(int(df["source_record"].iloc[-1]), 1712)
        self.assertEqual(df["source_file"].iloc[0], self.parquet_spec["file"])
    
    def test_map_canonical_columns(self) -> None:
        json_path = self.data_dir / self.json_spec["file"]
        df = read_json_records(
            file_path=json_path,
            expected_sha256 = self.json_spec["sha256"],
            expected_bytes = self.json_spec["bytes"],
        )
        mapped = map_canonical_columns(
            df = df,
            column_mapping = self.json_spec["column_mapping"],
            required_columns = self.canonical_cols,
        )
        for col in self.canonical_cols:
            self.assertIn(col, mapped.columns)

    def test_map_canonical_missing_columns(self) -> None:
        json_path = self.data_dir / self.json_spec["file"]
        df = read_json_records(file_path = json_path)
        bad_mapping = {}
        with self.assertRaises(FileFormatError):
            map_canonical_columns(
                df = df,
                column_mapping = bad_mapping,
                required_columns = self.canonical_cols,
            )


if __name__ == "__main__":
    unittest.main()
