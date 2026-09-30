#!/usr/bin/env python3
"""
merge_feature_datasets.py

Merge:
  1) q_features_*.csv
  2) dcir_features_*.csv
  3) time_features_*.csv

using the ONLY valid join key:
    (cell_global_id, rpt_id)

The output intentionally drops extraction/QC/debug columns and keeps only:
- minimal sample identity
- model candidate features
- SOH target

Important:
- Row order is NEVER used for matching.
- Missing values are NEVER filled with 0.
- By default, rows are filtered to aging_temperature_C == 25.0.
  This also protects against accidental matches such as folder "25. 45°C #1"
  when an earlier regex filter was simply "25".
- "broad" in an input filename has no special meaning and is ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


KEYS = ["cell_global_id", "rpt_id"]

Q_FEATURES = [
    "var_dq_3p3_4p2_Ah2",
    "mean_dq_gt_3p8_Ah",
    "mean_dq_3p8_3p9_Ah",
    "mean_dq_3p9_4p0_Ah",
    "mean_dq_4p0_4p1_Ah",
    "mean_dq_4p1_4p2_Ah",
    "max_dq_gt_3p8_Ah",
    "min_dq_gt_3p8_Ah",
    "range_dq_gt_3p8_Ah",
    "std_dq_gt_3p8_Ah",
]

# Hardware-aligned SOC50 3-A pulse feature + SOC 0~80% window-mean DCIRs.
DCIR_FEATURES = [
    "dcir_project_ohm",
    "dcir_soc00_window_mean_ohm",
    "dcir_soc10_window_mean_ohm",
    "dcir_soc20_window_mean_ohm",
    "dcir_soc30_window_mean_ohm",
    "dcir_soc40_window_mean_ohm",
    "dcir_soc50_window_mean_ohm",
    "dcir_soc60_window_mean_ohm",
    "dcir_soc70_window_mean_ohm",
    "dcir_soc80_window_mean_ohm",
]

TIME_FEATURES = [
    "delta_t_0p2c_3p8_4p0_s",
    "delta_t_1c_3p8_4p0_s",
    "charge_time_ratio_1c_over_0p2c",
    "cc_charge_time_s",
    "cv_charge_time_s",
]

TARGET = "soh_label_pct"

OUTPUT_COLUMNS = (
    ["cell_global_id", "rpt_id", "aging_condition"]
    + Q_FEATURES
    + DCIR_FEATURES
    + TIME_FEATURES
    + [TARGET]
)


class MergeError(RuntimeError):
    pass


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            raise MergeError(f"{path}: CSV header not found.")
    return rows


def is_blank(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def as_int(value, name: str) -> int:
    if is_blank(value):
        raise MergeError(f"Missing {name}.")
    try:
        return int(float(str(value).strip()))
    except Exception as e:
        raise MergeError(f"Invalid {name}: {value!r}") from e


def as_float(value, name: str) -> float:
    if is_blank(value):
        raise MergeError(f"Missing {name}.")
    try:
        x = float(str(value).strip())
    except Exception as e:
        raise MergeError(f"Invalid {name}: {value!r}") from e
    if not math.isfinite(x):
        raise MergeError(f"Non-finite {name}: {value!r}")
    return x


def require_columns(rows: list[dict], required: list[str], label: str):
    if not rows:
        raise MergeError(f"{label}: no rows.")
    present = set(rows[0].keys())
    missing = [c for c in required if c not in present]
    if missing:
        raise MergeError(f"{label}: missing columns: {missing}")


def filter_temperature(rows: list[dict], temp_c: float | None, label: str) -> list[dict]:
    if temp_c is None:
        return rows

    if "aging_temperature_C" not in rows[0]:
        raise MergeError(
            f"{label}: aging_temperature_C is required for temperature filtering."
        )

    kept = []
    removed = 0

    for row in rows:
        try:
            t = as_float(row.get("aging_temperature_C"), "aging_temperature_C")
        except MergeError:
            removed += 1
            continue

        if math.isclose(t, temp_c, rel_tol=0.0, abs_tol=1e-9):
            kept.append(row)
        else:
            removed += 1

    print(f"{label}: temperature filter {temp_c:g}°C -> {len(kept)} kept, {removed} removed")
    return kept


def validate_status(rows: list[dict], label: str):
    if "status" not in rows[0]:
        return
    bad = [r for r in rows if str(r.get("status", "")).strip().upper() != "OK"]
    if bad:
        raise MergeError(f"{label}: {len(bad)} rows have status != OK.")


def validate_constant(rows: list[dict], columns: list[str], label: str):
    for col in columns:
        if col not in rows[0]:
            continue
        vals = {str(r.get(col, "")).strip() for r in rows}
        if len(vals) > 1:
            raise MergeError(
                f"{label}: mixed definitions found in column {col}: {sorted(vals)}"
            )
        if vals:
            print(f"{label}: {col} = {next(iter(vals))}")


def build_index(rows: list[dict], label: str) -> dict[tuple[int, int], dict]:
    out = {}

    for row in rows:
        key = (
            as_int(row.get("cell_global_id"), "cell_global_id"),
            as_int(row.get("rpt_id"), "rpt_id"),
        )

        if key in out:
            raise MergeError(
                f"{label}: duplicate key (cell_global_id, rpt_id) = {key}"
            )

        out[key] = row

    return out


def validate_numeric_features(row: dict, columns: list[str], label: str, key):
    for col in columns:
        as_float(row.get(col), f"{label}.{col} at key={key}")


def write_csv(path: str | Path, rows: list[dict]):
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Merge Q, DCIR, and time features by cell_global_id + rpt_id."
    )

    ap.add_argument("--q", required=True, help="q_features CSV")
    ap.add_argument("--dcir", required=True, help="dcir_features CSV")
    ap.add_argument("--time", required=True, help="time_features CSV")
    ap.add_argument("-o", "--output", default="merged_features_25C.csv")

    ap.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Keep this aging temperature only. Default: 25.0",
    )

    ap.add_argument(
        "--strict-keys",
        action="store_true",
        help="Fail if the three files do not contain exactly the same keys.",
    )

    args = ap.parse_args()

    q_rows = read_csv(args.q)
    d_rows = read_csv(args.dcir)
    t_rows = read_csv(args.time)

    require_columns(
        q_rows,
        KEYS + ["aging_condition", "aging_temperature_C", "status"]
        + Q_FEATURES + [TARGET],
        "Q",
    )
    require_columns(
        d_rows,
        KEYS + ["aging_condition", "aging_temperature_C", "status"]
        + DCIR_FEATURES,
        "DCIR",
    )
    require_columns(
        t_rows,
        KEYS + ["aging_condition", "aging_temperature_C", "status"]
        + TIME_FEATURES,
        "TIME",
    )

    # Exact temperature filtering, not filename/regex filtering.
    q_rows = filter_temperature(q_rows, args.temperature, "Q")
    d_rows = filter_temperature(d_rows, args.temperature, "DCIR")
    t_rows = filter_temperature(t_rows, args.temperature, "TIME")

    validate_status(q_rows, "Q")
    validate_status(d_rows, "DCIR")
    validate_status(t_rows, "TIME")

    # Ensure all rows use the same extraction definitions before those
    # metadata columns are intentionally dropped from the final training CSV.
    validate_constant(
        d_rows,
        ["window_start_s", "window_end_s", "project_mode"],
        "DCIR",
    )
    validate_constant(
        t_rows,
        ["cccv_rate_used"],
        "TIME",
    )

    qi = build_index(q_rows, "Q")
    di = build_index(d_rows, "DCIR")
    ti = build_index(t_rows, "TIME")

    qk, dk, tk = set(qi), set(di), set(ti)
    common = qk & dk & tk

    print(f"Q keys:     {len(qk)}")
    print(f"DCIR keys:  {len(dk)}")
    print(f"TIME keys:  {len(tk)}")
    print(f"Common keys:{len(common)}")

    if args.strict_keys and not (qk == dk == tk):
        raise MergeError(
            "Key sets differ. "
            f"Q-only={len(qk - dk - tk)}, "
            f"DCIR-only={len(dk - qk - tk)}, "
            f"TIME-only={len(tk - qk - dk)}"
        )

    if not common:
        raise MergeError("No common (cell_global_id, rpt_id) keys.")

    merged = []

    for key in sorted(common):
        q = qi[key]
        d = di[key]
        t = ti[key]

        # Same key must also describe the same aging condition.
        conds = {
            str(q.get("aging_condition", "")).strip(),
            str(d.get("aging_condition", "")).strip(),
            str(t.get("aging_condition", "")).strip(),
        }

        if len(conds) != 1:
            raise MergeError(
                f"Aging-condition mismatch at key={key}: {sorted(conds)}"
            )

        validate_numeric_features(q, Q_FEATURES + [TARGET], "Q", key)
        validate_numeric_features(d, DCIR_FEATURES, "DCIR", key)
        validate_numeric_features(t, TIME_FEATURES, "TIME", key)

        row = {
            "cell_global_id": key[0],
            "rpt_id": key[1],
            "aging_condition": next(iter(conds)),
        }

        for col in Q_FEATURES:
            row[col] = q[col]

        for col in DCIR_FEATURES:
            row[col] = d[col]

        for col in TIME_FEATURES:
            row[col] = t[col]

        row[TARGET] = q[TARGET]
        merged.append(row)

    write_csv(args.output, merged)

    print(f"Output rows: {len(merged)}")
    print(f"Output columns: {len(OUTPUT_COLUMNS)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
