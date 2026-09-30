#!/usr/bin/env python3
"""
q_feature_extractor_v2.py

Folder-aware Q-statistics extractor for the Samsung INR18650-30Q RPT dataset.

Expected source layout
----------------------
Battery raw data/
└─ RPT Data/
   ├─ 1. LOW C@ 25°C #1/
   │  ├─ 0.csv
   │  ├─ 1.csv
   │  └─ ...
   ├─ 6. LOW C@ 25°C #6/
   │  ├─ 0.csv
   │  ├─ 1.csv
   │  └─ ...
   └─ ...

The script intentionally uses ONLY the RPT Data tree for ΔQ/Q-statistics.
Cycle Data, After SOH80 Data, and 0.0033C RPT Data are different experiments
and are not mixed into the ΔQ training rows.

One accepted output row = one (cell, RPT) diagnostic point.

Q feature definition
--------------------
ΔQ(V) = Q_0.2C(V) - Q_1C(V)

0.2C = 0.6 A for a 3 Ah cell
1C   = 3.0 A for a 3 Ah cell

Both curves are interpolated onto 1000 evenly spaced points over 3.000–4.195 V.

Features:
- Var(ΔQ), V >= 3.3 V
- Mean(ΔQ), V > 3.8 V
- Mean(ΔQ) in 3.8–3.9, 3.9–4.0, 4.0–4.1, 4.1–4.195 V
- Max / Min / Range / Std(ΔQ), V > 3.8 V

Protocol/QC rules
-----------------
- Prefer the SECOND complete 0.2C and 1C charge repetition (CurCycle=2),
  matching the repeated RPT protocol and the supplied processed reference files.
- If only one complete repetition exists, accept it but mark a QC warning.
- Reject short 3 A DCIR pulses: a Q curve must have sufficient voltage coverage,
  duration, and capacity span.
- Never fill a missing curve/feature with zero.
- RPT filename (0.csv, 1.csv, ...) is treated as RPT index.
- Parent folder name is treated as cell/aging-history metadata, NOT a model feature.
- TotCycle/CurCycle inside an RPT file are protocol-internal counters, not battery
  lifetime cycle count.

Optional SOH label support
--------------------------
The second complete 0.2C discharge capacity is retained as metadata. In batch mode,
SOH_label_pct is computed per cell relative to that cell's RPT0 second-0.2C
discharge capacity. This label is NOT an input feature.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.interpolate import interp1d


# -------------------- Project constants --------------------
NOMINAL_CAPACITY_AH = 3.0
TARGET_02C_A = 0.2 * NOMINAL_CAPACITY_AH   # 0.6 A
TARGET_1C_A  = 1.0 * NOMINAL_CAPACITY_AH   # 3.0 A

GRID_V_MIN = 3.000
GRID_V_MAX = 4.195
GRID_POINTS = 1000

VAR_V_MIN = 3.3
HIGH_V_MIN = 3.8

# Full-charge QC: designed to reject 10 s DCIR pulses.
CURRENT_REL_TOL = 0.12
CURRENT_ABS_TOL_MIN = 0.05
MIN_FULL_CHARGE_DURATION_S = 300.0
MIN_FULL_CHARGE_QSPAN_AH = 1.0
MIN_SEGMENT_ROWS = 20
LOWER_COVERAGE_MARGIN_V = 0.005
UPPER_COVERAGE_MARGIN_V = 0.002

# Preferred protocol steps in the supplied RPT files.
PREFERRED_02C_STEP = "6"
PREFERRED_1C_STEP = "11"
PREFERRED_REPEAT = "2"

# 0.2C discharge used as SOH ground-truth capacity metadata.
PREFERRED_02C_DISCHARGE_STEP = "7"


class QFeatureError(RuntimeError):
    pass


def _f(value) -> float:
    try:
        s = str(value).strip()
        return float(s) if s else math.nan
    except Exception:
        return math.nan


def parse_hms_seconds(value) -> float:
    """Parse H:M:S(.xx), M:S, or numeric seconds."""
    if value is None:
        return math.nan
    s = str(value).strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except Exception:
        pass

    parts = s.split(":")
    try:
        nums = [float(x) for x in parts]
    except Exception:
        return math.nan

    if len(nums) == 3:
        h, m, sec = nums
        return h * 3600.0 + m * 60.0 + sec
    if len(nums) == 2:
        m, sec = nums
        return m * 60.0 + sec
    return math.nan


def read_raw_csv(path: str | Path) -> List[dict]:
    """
    UTF-8-sig is the normal case. errors='replace' tolerates a corrupted
    non-English terminal/status string without damaging numeric columns.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"StepNo", "Type", "Voltage(V)", "Current(A)"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise QFeatureError(f"Missing required columns: {sorted(missing)}")
    return rows


def contiguous_segments(rows: List[dict]) -> List[Tuple[tuple, List[dict]]]:
    """
    Split on protocol identity. This prevents equal StepNo values from different
    repeats from being merged.
    """
    out = []
    current_key = None
    buf = []

    for row in rows:
        key = (
            str(row.get("StepNo", "")).strip(),
            str(row.get("Type", "")).strip().lower(),
            str(row.get("CurCycle", "")).strip(),
            str(row.get("TotCycle", "")).strip(),
        )
        if current_key is None:
            current_key = key

        if key != current_key:
            out.append((current_key, buf))
            current_key = key
            buf = []
        buf.append(row)

    if buf:
        out.append((current_key, buf))
    return out


def _capacity_series(rows: List[dict]) -> np.ndarray:
    """
    Prefer the cycler's Capacity(Ah). If it is unavailable, fall back to
    coulomb counting from Current(A) and StepTime.
    """
    q = np.array([_f(r.get("Capacity(Ah)", "")) for r in rows], dtype=float)
    if np.sum(np.isfinite(q)) >= max(3, int(0.8 * len(rows))):
        # Remove any step-level offset while preserving the original shape.
        finite_idx = np.where(np.isfinite(q))[0]
        q0 = q[finite_idx[0]] if len(finite_idx) else 0.0
        return q - q0

    current = np.array([_f(r.get("Current(A)", "")) for r in rows], dtype=float)
    t = np.array([parse_hms_seconds(r.get("StepTime(H:M:S)", "")) for r in rows], dtype=float)

    if np.sum(np.isfinite(current) & np.isfinite(t)) < 3:
        return q

    out = np.zeros(len(rows), dtype=float)
    for i in range(1, len(rows)):
        if all(map(math.isfinite, [current[i-1], current[i], t[i-1], t[i]])):
            dt = max(0.0, t[i] - t[i-1])
            # Trapezoidal coulomb counting; charge magnitude in Ah.
            out[i] = out[i-1] + max(0.0, 0.5 * (current[i-1] + current[i])) * dt / 3600.0
        else:
            out[i] = out[i-1]
    return out


def segment_summary(key: tuple, rows: List[dict]) -> dict:
    V = np.array([_f(r.get("Voltage(V)", "")) for r in rows], dtype=float)
    I = np.array([_f(r.get("Current(A)", "")) for r in rows], dtype=float)
    Q = _capacity_series(rows)
    T = np.array([parse_hms_seconds(r.get("StepTime(H:M:S)", "")) for r in rows], dtype=float)

    valid_vi = np.isfinite(V) & np.isfinite(I)
    if not np.any(valid_vi):
        return {
            "key": key, "rows": rows, "n": 0,
            "vmin": math.nan, "vmax": math.nan, "inom": math.nan,
            "qspan": math.nan, "duration_s": math.nan,
        }

    # Estimate CC current below the high-voltage CV region.
    body = valid_vi & (I > 0.05) & (V <= 4.15)
    if not np.any(body):
        body = valid_vi & (I > 0.05)
    inom = float(np.median(I[body])) if np.any(body) else math.nan

    qfinite = Q[np.isfinite(Q)]
    qspan = float(np.nanmax(qfinite) - np.nanmin(qfinite)) if len(qfinite) else math.nan
    tfinite = T[np.isfinite(T)]
    duration = float(np.nanmax(tfinite) - np.nanmin(tfinite)) if len(tfinite) else math.nan

    return {
        "key": key,
        "rows": rows,
        "n": int(np.sum(valid_vi)),
        "vmin": float(np.nanmin(V[valid_vi])),
        "vmax": float(np.nanmax(V[valid_vi])),
        "inom": inom,
        "qspan": qspan,
        "duration_s": duration,
    }


def is_complete_charge(info: dict, target_current_a: float) -> bool:
    tol = max(CURRENT_ABS_TOL_MIN, abs(target_current_a) * CURRENT_REL_TOL)
    return (
        info["n"] >= MIN_SEGMENT_ROWS
        and math.isfinite(info["inom"])
        and abs(info["inom"] - target_current_a) <= tol
        and math.isfinite(info["vmin"])
        and info["vmin"] <= GRID_V_MIN + LOWER_COVERAGE_MARGIN_V
        and math.isfinite(info["vmax"])
        and info["vmax"] >= GRID_V_MAX - UPPER_COVERAGE_MARGIN_V
        and math.isfinite(info["qspan"])
        and info["qspan"] >= MIN_FULL_CHARGE_QSPAN_AH
        and (
            not math.isfinite(info["duration_s"])
            or info["duration_s"] >= MIN_FULL_CHARGE_DURATION_S
        )
    )


def find_complete_charge(
    rows: List[dict],
    target_current_a: float,
    preferred_step: str | None = None,
) -> tuple[dict, List[dict], str]:
    """
    Prefer the second complete protocol repetition. Otherwise use the latest
    complete repetition and mark a warning.
    """
    candidates = []

    for key, seg_rows in contiguous_segments(rows):
        step_no, typ, cur_cycle, tot_cycle = key
        if typ != "charge":
            continue

        info = segment_summary(key, seg_rows)
        if is_complete_charge(info, target_current_a):
            candidates.append(info)

    if not candidates:
        raise QFeatureError(
            f"No complete {target_current_a:.3g} A charge curve. "
            "Short/partial pulses are not accepted as Q(V)."
        )

    # First choice: expected StepNo + second repeat.
    if preferred_step is not None:
        exact = [
            c for c in candidates
            if c["key"][0] == str(preferred_step) and c["key"][2] == PREFERRED_REPEAT
        ]
        if exact:
            return exact[-1], candidates, "protocol_second_repeat"

    # Second choice: any second repeat at the right current.
    second = [c for c in candidates if c["key"][2] == PREFERRED_REPEAT]
    if second:
        return second[-1], candidates, "second_repeat_fallback"

    # Final fallback: latest complete curve.
    return candidates[-1], candidates, "single_or_nonstandard_repeat"


def find_second_02c_discharge_capacity(rows: List[dict]) -> float:
    """
    Return second complete 0.2C discharge capacity for SOH label metadata.
    This does NOT become an input feature.
    """
    candidates = []
    for key, seg_rows in contiguous_segments(rows):
        step_no, typ, cur_cycle, tot_cycle = key
        if typ != "discharge":
            continue

        V = np.array([_f(r.get("Voltage(V)", "")) for r in seg_rows], dtype=float)
        I = np.array([_f(r.get("Current(A)", "")) for r in seg_rows], dtype=float)
        Q = _capacity_series(seg_rows)
        valid = np.isfinite(V) & np.isfinite(I)

        if not np.any(valid):
            continue

        body = valid & (I < -0.05)
        if not np.any(body):
            continue
        inom = float(np.median(np.abs(I[body])))
        tol = max(CURRENT_ABS_TOL_MIN, TARGET_02C_A * CURRENT_REL_TOL)

        qfinite = Q[np.isfinite(Q)]
        qspan = float(np.nanmax(qfinite) - np.nanmin(qfinite)) if len(qfinite) else math.nan

        if (
            abs(inom - TARGET_02C_A) <= tol
            and np.nanmax(V[valid]) >= 4.15
            and np.nanmin(V[valid]) <= 2.55
            and math.isfinite(qspan)
            and qspan >= 1.0
        ):
            candidates.append((key, qspan))

    # Prefer expected Step 7 / CurCycle 2.
    exact = [
        q for key, q in candidates
        if key[0] == PREFERRED_02C_DISCHARGE_STEP and key[2] == PREFERRED_REPEAT
    ]
    if exact:
        return float(exact[-1])

    second = [q for key, q in candidates if key[2] == PREFERRED_REPEAT]
    if second:
        return float(second[-1])

    return float(candidates[-1][1]) if candidates else math.nan


def prepare_qv_curve(segment: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use chronological data from segment start through the first sample reaching
    4.195 V. Then make voltage strictly increasing for interpolation.
    """
    rows = segment["rows"]
    q_all = _capacity_series(rows)
    points = []

    for idx, row in enumerate(rows):
        v = _f(row.get("Voltage(V)", ""))
        q = q_all[idx] if idx < len(q_all) else math.nan
        if not (math.isfinite(v) and math.isfinite(q)):
            continue
        points.append((v, q))
        if v >= GRID_V_MAX:
            break

    if len(points) < 3:
        raise QFeatureError("Too few valid Q-V points.")

    v = np.array([x[0] for x in points], dtype=float)
    q = np.array([x[1] for x in points], dtype=float)

    if np.nanmin(v) > GRID_V_MIN + LOWER_COVERAGE_MARGIN_V:
        raise QFeatureError("Selected curve does not cover the 3.0 V interpolation start.")
    if np.nanmax(v) < GRID_V_MAX - UPPER_COVERAGE_MARGIN_V:
        raise QFeatureError("Selected curve does not reach 4.195 V.")

    order = np.argsort(v, kind="stable")
    v, q = v[order], q[order]

    unique_v, inverse = np.unique(v, return_inverse=True)
    unique_q = np.array(
        [q[inverse == k].mean() for k in range(len(unique_v))],
        dtype=float,
    )

    if len(unique_v) < 3:
        raise QFeatureError("Too few unique voltages for interpolation.")

    return unique_v, unique_q


def parse_channel_number(channel: str) -> int | None:
    m = re.search(r"Ch\s*0*(\d+)", str(channel), flags=re.I)
    return int(m.group(1)) if m else None


def parse_cell_folder(folder_name: str) -> dict:
    """
    Example:
      '6. LOW C@ 25°C #6'
    """
    raw = str(folder_name).strip()
    m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", raw)
    global_id = int(m.group(1)) if m else None
    label = m.group(2).strip() if m else raw

    temp = None
    tm = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*C", label, flags=re.I)
    if tm:
        try:
            temp = float(tm.group(1))
        except Exception:
            pass

    return {
        "cell_global_id": global_id,
        "cell_folder": raw,
        "aging_condition": label,
        "aging_temperature_C": temp,
    }


def parse_rpt_id(path: Path) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*", path.stem)
    return int(m.group(1)) if m else None


def extract_q_statistics(path: str | Path, metadata: dict | None = None, return_curve: bool = False):
    path = Path(path)
    rows = read_raw_csv(path)

    seg02, cands02, sel02 = find_complete_charge(
        rows, TARGET_02C_A, PREFERRED_02C_STEP
    )
    seg1, cands1, sel1 = find_complete_charge(
        rows, TARGET_1C_A, PREFERRED_1C_STEP
    )

    v02, q02 = prepare_qv_curve(seg02)
    v1, q1 = prepare_qv_curve(seg1)

    voltage = np.linspace(GRID_V_MIN, GRID_V_MAX, GRID_POINTS)
    q02i = interp1d(v02, q02, kind="quadratic", bounds_error=True)(voltage)
    q1i = interp1d(v1, q1, kind="quadratic", bounds_error=True)(voltage)
    delta_q = q02i - q1i

    var_mask = voltage >= VAR_V_MIN
    high_mask = voltage > HIGH_V_MIN

    windows = {
        "mean_dq_3p8_3p9_Ah": (voltage >= 3.8) & (voltage < 3.9),
        "mean_dq_3p9_4p0_Ah": (voltage >= 3.9) & (voltage < 4.0),
        "mean_dq_4p0_4p1_Ah": (voltage >= 4.0) & (voltage < 4.1),
        # Paper interpolation stops at 4.195 V, so the intended 4.1–4.2 V
        # feature is represented by 4.1–4.195 V.
        "mean_dq_4p1_4p2_Ah": (voltage >= 4.1) & (voltage <= GRID_V_MAX),
    }

    high = delta_q[high_mask]

    features = {
        "var_dq_3p3_4p2_Ah2": float(np.var(delta_q[var_mask], ddof=1)),
        "mean_dq_gt_3p8_Ah": float(np.mean(high)),
        **{name: float(np.mean(delta_q[mask])) for name, mask in windows.items()},
        "max_dq_gt_3p8_Ah": float(np.max(high)),
        "min_dq_gt_3p8_Ah": float(np.min(high)),
        "range_dq_gt_3p8_Ah": float(np.max(high) - np.min(high)),
        "std_dq_gt_3p8_Ah": float(np.std(high, ddof=1)),
    }

    channel = str(rows[0].get("Channel", "")).strip() if rows else ""
    channel_num = parse_channel_number(channel)
    discharge_02c = find_second_02c_discharge_capacity(rows)

    qc = {
        "source_file": str(path),
        "channel": channel,
        "channel_num": channel_num,
        "rpt_id": parse_rpt_id(path),

        "q02_selection": sel02,
        "q02_step": seg02["key"][0],
        "q02_repeat": seg02["key"][2],
        "q02_current_A": seg02["inom"],
        "q02_duration_s": seg02["duration_s"],
        "q02_capacity_span_Ah": seg02["qspan"],
        "q02_complete_candidates": len(cands02),

        "q1_selection": sel1,
        "q1_step": seg1["key"][0],
        "q1_repeat": seg1["key"][2],
        "q1_current_A": seg1["inom"],
        "q1_duration_s": seg1["duration_s"],
        "q1_capacity_span_Ah": seg1["qspan"],
        "q1_complete_candidates": len(cands1),

        "discharge_0p2C_capacity_Ah": discharge_02c,
        "delta_q_at_4p195_Ah": float(delta_q[-1]),
        "status": "OK",
    }

    if metadata:
        qc = {**metadata, **qc}

    # Cross-check folder global cell id against cycler channel when possible.
    folder_id = qc.get("cell_global_id")
    if folder_id is not None and channel_num is not None:
        qc["cell_channel_match"] = bool(folder_id == channel_num)
    else:
        qc["cell_channel_match"] = None

    # A warning is metadata, not rejection.
    warnings = []
    if sel02 == "single_or_nonstandard_repeat":
        warnings.append("0.2C second repeat unavailable")
    if sel1 == "single_or_nonstandard_repeat":
        warnings.append("1C second repeat unavailable")
    if qc.get("cell_channel_match") is False:
        warnings.append("folder cell id != channel number")
    qc["qc_warning"] = "; ".join(warnings)

    if return_curve:
        curve = {
            "voltage_V": voltage,
            "q_0p2C_Ah": np.asarray(q02i, float),
            "q_1C_Ah": np.asarray(q1i, float),
            "delta_q_Ah": np.asarray(delta_q, float),
        }
        return features, qc, curve

    return features, qc


def resolve_rpt_root(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p

    if p.name.strip().lower() == "rpt data":
        return p

    child = p / "RPT Data"
    if child.is_dir():
        return child

    raise QFeatureError(
        "Directory must be the 'RPT Data' folder or its parent 'Battery raw data' folder."
    )


def discover_rpt_csvs(root: str | Path) -> List[tuple[Path, dict]]:
    p = resolve_rpt_root(root)

    if p.is_file():
        return [(p, {})]

    found = []
    for cell_dir in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name):
        meta = parse_cell_folder(cell_dir.name)

        for f in cell_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".csv", ".txt"}:
                continue
            rpt_id = parse_rpt_id(f)
            if rpt_id is None:
                # Numeric filenames are the RPT points in the shown folder structure.
                continue

            found.append((f, {**meta, "rpt_id_from_filename": rpt_id}))

    found.sort(
        key=lambda item: (
            item[1].get("cell_global_id") if item[1].get("cell_global_id") is not None else 10**9,
            item[1].get("rpt_id_from_filename", 10**9),
            str(item[0]),
        )
    )
    return found


def extract_rpt_tree(
    root: str | Path,
    include_regex: str | None = None,
) -> tuple[List[dict], List[dict]]:
    """
    Process all numeric RPT CSV files under RPT Data/<cell folder>/.
    include_regex filters CELL FOLDER NAMES only, if explicitly requested.
    """
    items = discover_rpt_csvs(root)
    good = []
    bad = []
    pattern = re.compile(include_regex, re.I) if include_regex else None

    for path, meta in items:
        if pattern and not pattern.search(meta.get("cell_folder", "")):
            continue

        try:
            features, qc = extract_q_statistics(path, metadata=meta)
            # Use numeric filename as authoritative RPT id in folder mode.
            if meta.get("rpt_id_from_filename") is not None:
                qc["rpt_id"] = meta["rpt_id_from_filename"]
            good.append({**qc, **features})
        except Exception as e:
            bad.append({
                **meta,
                "source_file": str(path),
                "rpt_id": meta.get("rpt_id_from_filename"),
                "status": "REJECTED",
                "reason": str(e),
            })

    # Build SOH label from RPT0 0.2C discharge for each cell.
    baselines = {}
    for row in good:
        cid = row.get("cell_global_id")
        rpt = row.get("rpt_id")
        cap = row.get("discharge_0p2C_capacity_Ah")
        if cid is not None and rpt == 0 and isinstance(cap, (int, float)) and math.isfinite(cap) and cap > 0:
            baselines[cid] = cap

    for row in good:
        cid = row.get("cell_global_id")
        cap = row.get("discharge_0p2C_capacity_Ah")
        baseline = baselines.get(cid)
        row["soh_label_pct"] = (
            float(cap / baseline * 100.0)
            if baseline and isinstance(cap, (int, float)) and math.isfinite(cap)
            else math.nan
        )
        row["soh_label_source"] = "2nd 0.2C discharge / same-cell RPT0" if baseline else ""

    return good, bad


def write_csv(path: str | Path, rows: List[dict]):
    path = Path(path)
    if not rows:
        # Still create a file with no rows only when explicitly requested.
        path.write_text("", encoding="utf-8-sig")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Extract ΔQ/Q-statistics from the RPT Data directory tree."
    )
    ap.add_argument(
        "input",
        help="RPT Data folder, Battery raw data parent folder, or one RPT CSV."
    )
    ap.add_argument(
        "-o", "--output",
        default="q_features_all_rpt.csv"
    )
    ap.add_argument(
        "--rejected",
        default="q_features_rejected.csv"
    )
    ap.add_argument(
        "--include-regex",
        default=None,
        help=(
            "Optional regex applied to cell folder names. "
            "Example: 'LOW C@\\s*25' for a strict LOW-C 25C subset. "
            "Default: keep all RPT cells/aging histories."
        )
    )
    args = ap.parse_args()

    root = Path(args.input)
    if root.is_file():
        features, qc = extract_q_statistics(root)
        good = [{**qc, **features}]
        bad = []
    else:
        good, bad = extract_rpt_tree(root, args.include_regex)

    write_csv(args.output, good)
    write_csv(args.rejected, bad)

    print(f"accepted RPT samples: {len(good)}")
    print(f"rejected RPT samples: {len(bad)}")
    print(f"output: {args.output}")
    if bad:
        print(f"rejected log: {args.rejected}")


if __name__ == "__main__":
    main()
