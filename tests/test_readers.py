import unittest
from pathlib import Path
from src.readers import (
    FileFormatError,
    load_configuration,
    read_csv_records,
    read_excel_records,
    validate_input_file,
)


class TestMember1Readers(unittest.TestCase):
    """Automated checks for Member 1: CSV/Excel readers, config, and reader failure checks."""

    def setUp(self) -> None:
        self.config = load_configuration("config.json")
        self.data_dir = Path(self.config["data_dir"])
        self.csv_spec = self.config["file_specs"]["country_records.csv"]
        self.xlsx_spec = self.config["file_specs"]["country_records.xlsx"]

    def test_missing_file_raises_error(self) -> None:
        """Category 2: Missing file must stop processing."""
        missing_path = self.data_dir / "non_existent_file.csv"
        with self.assertRaises(FileNotFoundError):
            validate_input_file(missing_path)
        with self.assertRaises(FileNotFoundError):
            read_csv_records(missing_path)

    def test_unsupported_format_raises_error(self) -> None:
        """Category 3: Existing unsupported format (unsupported.txt) must stop processing."""
        unsupported_path = self.data_dir / "unsupported.txt"
        with self.assertRaises(FileFormatError):
            validate_input_file(unsupported_path)
        with self.assertRaises(FileFormatError):
            read_csv_records(unsupported_path)

    def test_csv_reader_recovers_1712_rows_and_drops_export_note(self) -> None:
        """Verify country_records.csv loads 1,712 rows, checks SHA256, and drops export_note."""
        csv_path = self.data_dir / self.csv_spec["file"]
        df = read_csv_records(
            file_path=csv_path,
            encoding=self.csv_spec["encoding"],
            separator=self.csv_spec["separator"],
            comment_prefix=self.csv_spec["comment_prefix"],
            ignore_columns=self.csv_spec["ignore_columns"],
            expected_sha256=self.csv_spec["sha256"],
            expected_bytes=self.csv_spec["bytes"],
        )
        self.assertEqual(len(df), 1712)
        self.assertNotIn("export_note", df.columns)
        self.assertEqual(int(df["source_record"].iloc[0]), 1)
        self.assertEqual(int(df["source_record"].iloc[-1]), 1712)

    def test_excel_reader_recovers_1712_rows_from_observations_sheet(self) -> None:
        """Verify country_records.xlsx skips Readme and loads 1,712 rows from observations."""
        xlsx_path = self.data_dir / self.xlsx_spec["file"]
        df = read_excel_records(
            file_path=xlsx_path,
            sheet_name=self.xlsx_spec["sheet_name"],
            engine=self.xlsx_spec["engine"],
            expected_sha256=self.xlsx_spec["sha256"],
            expected_bytes=self.xlsx_spec["bytes"],
        )
        self.assertEqual(len(df), 1712)
        self.assertEqual(int(df["source_record"].iloc[0]), 1)
        self.assertEqual(int(df["source_record"].iloc[-1]), 1712)

    def test_failure_csv_reader_configuration(self) -> None:
        """Verify failure-test CSVs (UTF-8, comma, no comments) can be parsed."""
        fail_cfg = self.config["failure_test_csv_settings"]
        missing_col_df = read_csv_records(
            file_path=self.data_dir / "missing_column.csv",
            encoding=fail_cfg["encoding"],
            separator=fail_cfg["separator"],
            comment_prefix=fail_cfg["comment_prefix"],
        )
        self.assertEqual(len(missing_col_df), 5)
        self.assertNotIn("pop", missing_col_df.columns)


if __name__ == "__main__":
    unittest.main()