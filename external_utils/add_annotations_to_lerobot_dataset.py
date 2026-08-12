#!/usr/bin/env python

"""Add a per-frame string annotation feature to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ANNOTATION_FEATURE = {
    "dtype": "string",
    "shape": [1],
    "names": None,
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_table_atomic(table: pa.Table, path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def find_data_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    paths = sorted(data_root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {data_root}")
    return paths


def add_annotation_column(
    dataset_root: str | Path,
    *,
    column: str = "annotation",
    default: str = "",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Add ``column`` to all dataset Parquet files and metadata.

    Existing columns are never overwritten. Returns ``(changed_files, total_files)``.
    """
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")

    info = load_json(info_path)
    existing_feature = info.get("features", {}).get(column)
    if existing_feature is not None and existing_feature.get("dtype") != "string":
        raise ValueError(
            f"Metadata feature {column!r} already exists with dtype "
            f"{existing_feature.get('dtype')!r}, expected 'string'"
        )

    data_paths = find_data_files(root)
    changed = 0
    for path in data_paths:
        schema = pq.read_schema(path)
        if column in schema.names:
            continue

        changed += 1
        if dry_run:
            continue

        table = pq.read_table(path)
        values = pa.array([default] * table.num_rows, type=pa.string())
        table = table.append_column(column, values)
        write_table_atomic(table, path)

    if not dry_run:
        missing = [path for path in data_paths if column not in pq.read_schema(path).names]
        if missing:
            preview = ", ".join(str(path.relative_to(root)) for path in missing[:5])
            raise RuntimeError(f"Annotation column is still missing from: {preview}")

        features = info.setdefault("features", {})
        features[column] = ANNOTATION_FEATURE.copy()
        write_json_atomic(info, info_path)

    return changed, len(data_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path containing data/, videos/, and meta/")
    parser.add_argument("--column", default="annotation", help="Annotation feature name")
    parser.add_argument("--default", default="", help="Initial value for unannotated frames")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changed, total = add_annotation_column(
        args.dataset_root,
        column=args.column,
        default=args.default,
        dry_run=args.dry_run,
    )
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {changed} of {total} Parquet files.")
    if not args.dry_run:
        print(f"Feature {args.column!r} is present in the dataset metadata and every data file.")


if __name__ == "__main__":
    main()
