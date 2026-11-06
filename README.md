## week10_Group-7
**DSA4153 Programming for Data Science: Week 10 Group Programming Challenge
Four-Format Data Recovery and Validation**

This program recovers the same clean Gapminder dataset from four differently formatted files (CSV, Excel, JSON, Parquet), checks that the four recoveries agree, produces pandas and NumPy results, exports the clean data, and verifies the exported files.

## Directory Structure

```
reiyonahraynes-week10_group-7/
├── README.md
├── config.json
├── contributions.csv
├── main.py
├── requirements.txt
├── data/
│   ├── README.txt
│   ├── manifest.json
│   ├── missing_column.csv
│   └── unsupported.txt
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── compare.py
│   ├── exports.py
│   ├── pipeline.py
│   ├── readers.py
│   ├── readers_json_parquet.py
│   └── validation.py
└── tests/
    ├── __init__.py
    ├── support.py
    ├── test_analysis.py
    ├── test_compare.py
    ├── test_exports.py
    ├── test_pipeline.py
    ├── test_readers.py
    ├── test_validation.py
    └── tests_json_parquet.py
```

## Setup

Requires **Python 3.10** or newer
