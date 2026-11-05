import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.pipeline import ProcessingFailure, ProcessingRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover and validate the country dataset.")
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument("--input", help="process a single file instead of the four formats")
    parser.add_argument("--year", type=int, help="override reporting_year from the config")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run = ProcessingRun(args.config)
    try:
        if args.year is not None:
            run.year_override = args.year
        summary = run.run_single_file(args.input) if args.input else run.run()
    except ProcessingFailure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        print("Nothing was published.", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected: still exit 1 and publish nothing
        print(f"FAILED: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    year = summary["reporting_year"]
    print(f"Success: {summary['canonical_rows']} canonical rows, {summary['total_quarantined']} quarantined, "
          f"{summary['total_duplicates_removed']} duplicates removed.")
    print(f"Asia {year}: {summary[f'asia_{year}_rows']} rows, population {summary[f'asia_{year}_population']:,}")
    print(f"Philippines {year}: {summary[f'philippines_{year}_population']:,}")
    print(f"Checks passed: {summary['checks_passed']}/{summary['checks_total']}")
    print(f"Outputs: {summary['published_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
