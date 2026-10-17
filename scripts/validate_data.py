"""Validate the three CSV files used by MMCA.

Run from the repository root with: python scripts/validate_data.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def feature_table(path: Path, id_column: str, expected_width: int) -> tuple[int, int, set[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        if not header or header[0] != id_column:
            raise ValueError(f"{path}: first column must be {id_column!r}")
        width = len(header) - 1
        if width != expected_width:
            raise ValueError(f"{path}: expected {expected_width} numeric columns, found {width}")
        identifiers: set[str] = set()
        count = 0
        for line_number, row in enumerate(rows, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line_number}: expected {len(header)} fields, found {len(row)}")
            identifier = row[0]
            if not identifier:
                raise ValueError(f"{path}:{line_number}: empty identifier")
            if identifier in identifiers:
                raise ValueError(f"{path}:{line_number}: duplicate identifier {identifier}")
            identifiers.add(identifier)
            for value in row[1:]:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{path}:{line_number}: non-finite feature value")
            count += 1
    return count, width, identifiers


def interactions(path: Path, drugs: set[str], proteins: set[str]) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["Drug", "Protein"]:
            raise ValueError(f"{path}: expected columns Drug,Protein")
        seen: set[tuple[str, str]] = set()
        count = 0
        for line_number, row in enumerate(rows, start=2):
            pair = (row["Drug"], row["Protein"])
            if pair in seen:
                raise ValueError(f"{path}:{line_number}: duplicate interaction {pair}")
            if pair[0] not in drugs:
                raise ValueError(f"{path}:{line_number}: unknown drug {pair[0]}")
            if pair[1] not in proteins:
                raise ValueError(f"{path}:{line_number}: unknown protein {pair[1]}")
            seen.add(pair)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data = args.root / "data"
    drug_count, drug_width, drugs = feature_table(data / "drug_feature.csv", "drug_name", 64)
    protein_count, protein_width, proteins = feature_table(data / "protein_feature.csv", "protein", 340)
    interaction_count = interactions(data / "drug_protein.csv", drugs, proteins)
    print(f"drug_feature.csv: {drug_count} rows x {drug_width} features")
    print(f"protein_feature.csv: {protein_count} rows x {protein_width} features")
    print(f"drug_protein.csv: {interaction_count} interactions")
    print("OK: schema, numeric values, identifiers, and interaction references are valid")


if __name__ == "__main__":
    main()
