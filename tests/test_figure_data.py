"""Check that every supplied figure table can be read with standard CSV tools."""

import csv
import re
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "figures"
TABLES = sorted(path for path in FIGURES.rglob("*.csv") if "results" not in path.parts)


@pytest.mark.parametrize("path", TABLES, ids=lambda path: path.name)
def test_supplied_csv_columns_and_code_labels(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames))
        rows = list(reader)
    assert rows
    for row in rows:
        assert None not in row, "Unquoted comma produced excess columns"
        assert all(value is not None for value in row.values()), "Missing CSV columns"
        if "parameter_label" in row:
            assert re.fullmatch(r"\[\[\d+,\d+,\d+\]\]", row["parameter_label"])
