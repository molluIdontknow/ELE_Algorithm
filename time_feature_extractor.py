#!/usr/bin/env python3
"""
time_feature_extractor.py

Time-feature extractor for the Samsung INR18650-30Q RPT raw CSV structure.

Features produced from one (cell, RPT):
1) Δt(0.2C, 3.8→4.0 V)
2) Δt(1C,   3.8→4.0 V)
3) Charge Time Ratio = Δt(1C) / Δt(0.2C)
4) CC Charge Time
5) CV Charge Time

The 0.2C and 1C charge curves are selected with the same philosophy as
q_feature_extractor_v2.py:
- 0.2C = ~0.6 A
- 1C   = ~3.0 A
- prefer the SECOND complete repetition (CurCycle=2)
- prefer Step 6 for 0.2C and Step 11 for 1C
- reject short/partial pulses
- never fill missing features with 0

CC/CV definition used here
--------------------------
The raw cycler file stores CC and CV in one charge StepNo. Therefore the script
must infer the CC→CV transition from V/I:

CV entry = first SUSTAINED high-voltage sample where
           V >= 4.19 V and I < 98% of the nominal CC current.

The next few samples must also stay below that threshold, which prevents one
noisy current sample from being interpreted as CV entry.

CV end = last active-current sample of the same charge step.
The supplied RPT sample ends around half of the CC current, matching the
CC-CV cutoff protocol.

Because the earlier feature list did not specify whether the single CC/CV pair
must come from 0.2C or 1C, BOTH rates are saved:
- cc_charge_time_0p2c_s / cv_charge_time_0p2c_s
- cc_charge_time_1c_s   / cv_charge_time_1c_s

For the final two canonical model columns:
- cc_charge_time_s
- cv_charge_time_s

choose the rate with --cccv-rate {1C,0.2C}. Default = 1C.
This choice is explicit metadata in cccv_rate_used, so the training dataset
cannot silently mix definitions.

Dependencies: numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -------------------- Cell / protocol constants --------------------
NOMINAL_CAPACITY_AH = 3.0
TARGET_02C_A = 0.2 * NOMINAL_CAPACITY_AH   # 0.6 A
TARGET_1C_A  = 1.0 * NOMINAL_CAPACITY_AH   # 3.0 A

PREFERRED_02C_STEP = "6"
PREFERRED_1C_STEP = "11"
PREFERRED_REPEAT = "2"

# Complete-charge selection
CURRENT_REL_TOL = 0.12
CURRENT_ABS_TOL_MIN = 0.05
MIN_FULL_CHARGE_DURATION_S = 300.0
MIN_SEGMENT_ROWS = 20
FULL_CURVE_V_START_MAX = 3.05
FULL_CURVE_V_END_MIN = 4.195

# Time-feature voltage interval
DT_V_START = 3.8
DT_V_END = 4.0

# CC→CV inference from raw V/I
CV_HIGH_VOLTAGE_MIN = 4.19
CV_CURRENT_FRACTION = 0.98
CV_SUSTAIN_SAMPLES = 3

# End-of-CV quality check. The supplied protocol finishes near 0.5×CC.
CUTOFF_CURRENT_FRACTION_MAX = 0.60
ACTIVE_CURRENT_MIN_A = 0.05


class TimeFeatureError(RuntimeError):
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
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {
        "StepNo",
        "Type",
        "CurCycle",
        "TotCycle",
        "StepTime(H:M:S)",
        "Voltage(V)",
        "Current(A)",
    }
    missing = required.difference(fieldnames)

    if missing:
        raise TimeFeatureError(
            f"Missing required columns: {sorted(missing)}"
        )

    return rows


def contiguous_segments(rows: List[dict]) -> List[Tuple[tuple, List[dict]]]:
    """
    Split whenever StepNo / Type / CurCycle / TotCycle changes.
    This prevents different protocol repetitions from being mixed.
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


def segment_arrays(rows: List[dict]):
    t = np.array(
        [parse_hms_seconds(r.get("StepTime(H:M:S)", "")) for r in rows],
        dtype=float,
    )
    v = np.array(
        [_f(r.get("Voltage(V)", "")) for r in rows],
        dtype=float,
    )
    i = np.array(
        [_f(r.get("Current(A)", "")) for r in rows],
        dtype=float,
    )

    good = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    t, v, i = t[good], v[good], i[good]

    if len(t) < 2:
        raise TimeFeatureError("Too few valid time/voltage/current rows.")

    order = np.argsort(t, kind="stable")
    return t[order], v[order], i[order]


def segment_summary(key: tuple, rows: List[dict]) -> dict:
    try:
        t, v, i = segment_arrays(rows)
    except Exception:
        return {
            "key": key,
            "rows": rows,
            "n": 0,
            "duration_s": math.nan,
            "vmin": math.nan,
            "vmax": math.nan,
            "inom": math.nan,
        }

    # Estimate CC current before the high-voltage CV region.
    body = (i > ACTIVE_CURRENT_MIN_A) & (v <= 4.15)
    if not np.any(body):
        body = i > ACTIVE_CURRENT_MIN_A

    inom = float(np.median(i[body])) if np.any(body) else math.nan

    return {
        "key": key,
        "rows": rows,
        "n": int(len(t)),
        "duration_s": float(np.max(t) - np.min(t)),
        "vmin": float(np.min(v)),
        "vmax": float(np.max(v)),
        "inom": inom,
    }


def is_complete_charge(info: dict, target_current_a: float) -> bool:
    tol = max(
        CURRENT_ABS_TOL_MIN,
        abs(target_current_a) * CURRENT_REL_TOL,
    )

    return (
        info["n"] >= MIN_SEGMENT_ROWS
        and math.isfinite(info["duration_s"])
        and info["duration_s"] >= MIN_FULL_CHARGE_DURATION_S
        and math.isfinite(info["vmin"])
        and info["vmin"] <= FULL_CURVE_V_START_MAX
        and math.isfinite(info["vmax"])
        and info["vmax"] >= FULL_CURVE_V_END_MIN
        and math.isfinite(info["inom"])
        and abs(info["inom"] - target_current_a) <= tol
    )


def find_complete_charge(
    rows: List[dict],
    target_current_a: float,
    preferred_step: str,
) -> tuple[dict, List[dict], str]:
    """
    Prefer:
      1) expected StepNo + CurCycle=2
      2) any complete CurCycle=2
      3) latest complete curve
    """
    candidates = []

    for key, seg_rows in contiguous_segments(rows):
        step_no, typ, curcycle, totcycle = key

        if typ != "charge":
            continue

        info = segment_summary(key, seg_rows)

        if is_complete_charge(info, target_current_a):
            candidates.append(info)

    if not candidates:
        raise TimeFeatureError(
            f"No complete {target_current_a:.3g} A charge curve found."
        )

    exact = [
        c for c in candidates
        if c["key"][0] == str(preferred_step)
        and c["key"][2] == PREFERRED_REPEAT
    ]
    if exact:
        return exact[-1], candidates, "protocol_second_repeat"

    second = [
        c for c in candidates
        if c["key"][2] == PREFERRED_REPEAT
    ]
    if second:
        return second[-1], candidates, "second_repeat_fallback"

    return (
        candidates[-1],
        candidates,
        "single_or_nonstandard_repeat",
    )


def first_upward_crossing_time(
    rows: List[dict],
    threshold_v: float,
) -> float:
    """
    Estimate the first upward voltage-crossing time by linear interpolation
    between adjacent raw samples.

    The difference between two crossing times is the Δt feature.
    """
    t, v, i = segment_arrays(rows)

    for j in range(1, len(v)):
        v0, v1 = v[j - 1], v[j]
        t0, t1 = t[j - 1], t[j]

        if v0 <= threshold_v <= v1 and v1 > v0:
            frac = (threshold_v - v0) / (v1 - v0)
            return float(t0 + frac * (t1 - t0))

    raise TimeFeatureError(
        f"Voltage never crosses {threshold_v:.3f} V upward."
    )


def voltage_window_time(rows: List[dict]) -> tuple[float, float, float]:
    t38 = first_upward_crossing_time(rows, DT_V_START)
    t40 = first_upward_crossing_time(rows, DT_V_END)

    dt = t40 - t38

    if not math.isfinite(dt) or dt <= 0:
        raise TimeFeatureError(
            f"Invalid {DT_V_START:.1f}→{DT_V_END:.1f} V passage time."
        )

    return t38, t40, dt


def infer_cc_cv_times(
    rows: List[dict],
    nominal_cc_a: float,
) -> dict:
    """
    Infer CC→CV transition from the combined raw charge step.

    CV entry:
      first high-voltage sample (V >= 4.19 V) whose current is below
      98% of nominal CC and stays below that threshold for several samples.

    This is deliberately based on current taper, rather than simply declaring
    4.195 V to be CV. In the supplied sample the cell reaches ~4.195 V while
    the current is still fully in CC mode.
    """
    t, v, i = segment_arrays(rows)

    threshold_i = CV_CURRENT_FRACTION * nominal_cc_a
    cv_idx = None

    for j in range(len(t)):
        if v[j] < CV_HIGH_VOLTAGE_MIN:
            continue
        if i[j] >= threshold_i:
            continue

        end = min(len(t), j + CV_SUSTAIN_SAMPLES)
        window_i = i[j:end]

        if len(window_i) < 2:
            continue

        # sustained taper: all following points remain below CC threshold
        if np.all(window_i < threshold_i):
            cv_idx = j
            break

    if cv_idx is None:
        raise TimeFeatureError(
            f"Could not infer CC→CV transition for nominal {nominal_cc_a:g} A."
        )

    # Charge starts at StepTime=0 in this cycler format. The t=0 record may
    # show I=0, but the next sample's accumulated capacity confirms that
    # current was applied during the interval immediately after t=0.
    charge_start_t = float(np.min(t))
    cv_start_t = float(t[cv_idx])

    active = np.where(i > ACTIVE_CURRENT_MIN_A)[0]
    if len(active) == 0:
        raise TimeFeatureError("No active charge current found.")

    end_idx = int(active[-1])
    charge_end_t = float(t[end_idx])
    cutoff_i = float(i[end_idx])

    # Check that the CV phase reached a reasonably low cutoff current.
    # Do not silently accept a truncated charge.
    if cutoff_i > CUTOFF_CURRENT_FRACTION_MAX * nominal_cc_a:
        raise TimeFeatureError(
            f"CV appears incomplete: final active current {cutoff_i:.4g} A "
            f"is above {CUTOFF_CURRENT_FRACTION_MAX:.0%} of "
            f"{nominal_cc_a:g} A."
        )

    cc_time = cv_start_t - charge_start_t
    cv_time = charge_end_t - cv_start_t

    if cc_time <= 0 or cv_time <= 0:
        raise TimeFeatureError("Invalid CC/CV duration.")

    return {
        "cc_time_s": float(cc_time),
        "cv_time_s": float(cv_time),
        "charge_start_time_s": charge_start_t,
        "cv_entry_time_s": cv_start_t,
        "charge_end_time_s": charge_end_t,
        "cv_entry_voltage_V": float(v[cv_idx]),
        "cv_entry_current_A": float(i[cv_idx]),
        "cutoff_current_A": cutoff_i,
        "cutoff_fraction_of_cc": float(cutoff_i / nominal_cc_a),
    }


def parse_channel_number(channel: str) -> int | None:
    m = re.search(r"Ch\s*0*(\d+)", str(channel), flags=re.I)
    return int(m.group(1)) if m else None


def parse_cell_folder(folder_name: str) -> dict:
    raw = str(folder_name).strip()

    m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", raw)
    global_id = int(m.group(1)) if m else None
    label = m.group(2).strip() if m else raw

    temp = None
    tm = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*C",
        label,
        flags=re.I,
    )
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


def extract_time_features(
    path: str | Path,
    cccv_rate: str = "1C",
    metadata: dict | None = None,
) -> dict:
    path = Path(path)
    rows = read_raw_csv(path)

    seg02, cands02, sel02 = find_complete_charge(
        rows,
        TARGET_02C_A,
        PREFERRED_02C_STEP,
    )
    seg1, cands1, sel1 = find_complete_charge(
        rows,
        TARGET_1C_A,
        PREFERRED_1C_STEP,
    )

    # Δt features
    t38_02, t40_02, dt02 = voltage_window_time(seg02["rows"])
    t38_1, t40_1, dt1 = voltage_window_time(seg1["rows"])

    ratio = dt1 / dt02
    if not math.isfinite(ratio) or ratio <= 0:
        raise TimeFeatureError("Invalid charge-time ratio.")

    # CC / CV features at BOTH C-rates
    cccv02 = infer_cc_cv_times(
        seg02["rows"],
        TARGET_02C_A,
    )
    cccv1 = infer_cc_cv_times(
        seg1["rows"],
        TARGET_1C_A,
    )

    if cccv_rate == "1C":
        canonical_cc = cccv1["cc_time_s"]
        canonical_cv = cccv1["cv_time_s"]
    elif cccv_rate == "0.2C":
        canonical_cc = cccv02["cc_time_s"]
        canonical_cv = cccv02["cv_time_s"]
    else:
        raise TimeFeatureError(
            "cccv_rate must be '1C' or '0.2C'."
        )

    channel = (
        str(rows[0].get("Channel", "")).strip()
        if rows else ""
    )
    channel_num = parse_channel_number(channel)

    out = {
        # Join keys / metadata
        "source_file": str(path),
        "channel": channel,
        "channel_num": channel_num,
        "rpt_id": parse_rpt_id(path),

        # -------- actual time features --------
        "delta_t_0p2c_3p8_4p0_s": dt02,
        "delta_t_1c_3p8_4p0_s": dt1,
        "charge_time_ratio_1c_over_0p2c": ratio,

        # Canonical final-model CC/CV columns
        "cccv_rate_used": cccv_rate,
        "cc_charge_time_s": canonical_cc,
        "cv_charge_time_s": canonical_cv,

        # Save both candidates so the definition can be changed later
        # without re-reading/reprocessing all raw data.
        "cc_charge_time_0p2c_s": cccv02["cc_time_s"],
        "cv_charge_time_0p2c_s": cccv02["cv_time_s"],
        "cc_charge_time_1c_s": cccv1["cc_time_s"],
        "cv_charge_time_1c_s": cccv1["cv_time_s"],

        # -------- QC / audit information --------
        "q02_selection": sel02,
        "q02_step": seg02["key"][0],
        "q02_repeat": seg02["key"][2],
        "q02_current_A": seg02["inom"],
        "q02_complete_candidates": len(cands02),

        "q1_selection": sel1,
        "q1_step": seg1["key"][0],
        "q1_repeat": seg1["key"][2],
        "q1_current_A": seg1["inom"],
        "q1_complete_candidates": len(cands1),

        "t_0p2c_cross_3p8_s": t38_02,
        "t_0p2c_cross_4p0_s": t40_02,
        "t_1c_cross_3p8_s": t38_1,
        "t_1c_cross_4p0_s": t40_1,

        "cv_entry_time_0p2c_s": cccv02["cv_entry_time_s"],
        "cv_entry_voltage_0p2c_V": cccv02["cv_entry_voltage_V"],
        "cv_entry_current_0p2c_A": cccv02["cv_entry_current_A"],
        "cutoff_current_0p2c_A": cccv02["cutoff_current_A"],

        "cv_entry_time_1c_s": cccv1["cv_entry_time_s"],
        "cv_entry_voltage_1c_V": cccv1["cv_entry_voltage_V"],
        "cv_entry_current_1c_A": cccv1["cv_entry_current_A"],
        "cutoff_current_1c_A": cccv1["cutoff_current_A"],

        "status": "OK",
    }

    if metadata:
        out = {**metadata, **out}

    folder_id = out.get("cell_global_id")

    if folder_id is not None and channel_num is not None:
        out["cell_channel_match"] = bool(
            int(folder_id) == int(channel_num)
        )
    else:
        out["cell_channel_match"] = None

    warnings = []

    if sel02 == "single_or_nonstandard_repeat":
        warnings.append("0.2C second repeat unavailable")

    if sel1 == "single_or_nonstandard_repeat":
        warnings.append("1C second repeat unavailable")

    if out.get("cell_channel_match") is False:
        warnings.append("folder cell id != channel number")

    out["qc_warning"] = "; ".join(warnings)

    return out


def resolve_rpt_root(path: str | Path) -> Path:
    p = Path(path)

    if p.is_file():
        return p

    if p.name.strip().lower() == "rpt data":
        return p

    child = p / "RPT Data"
    if child.is_dir():
        return child

    raise TimeFeatureError(
        "Input directory must be the 'RPT Data' folder or its parent "
        "'Battery raw data' folder."
    )


def discover_rpt_csvs(
    root: str | Path,
) -> List[tuple[Path, dict]]:
    p = resolve_rpt_root(root)

    if p.is_file():
        return [(p, {})]

    found = []

    for cell_dir in sorted(
        [x for x in p.iterdir() if x.is_dir()],
        key=lambda x: x.name,
    ):
        meta = parse_cell_folder(cell_dir.name)

        for f in cell_dir.iterdir():
            if (
                not f.is_file()
                or f.suffix.lower() not in {".csv", ".txt"}
            ):
                continue

            rpt_id = parse_rpt_id(f)

            # In the shown folder structure, numeric filenames are RPT ids.
            if rpt_id is None:
                continue

            found.append(
                (
                    f,
                    {
                        **meta,
                        "rpt_id_from_filename": rpt_id,
                    },
                )
            )

    found.sort(
        key=lambda item: (
            item[1].get("cell_global_id")
            if item[1].get("cell_global_id") is not None
            else 10**9,
            item[1].get("rpt_id_from_filename", 10**9),
            str(item[0]),
        )
    )

    return found


def extract_rpt_tree(
    root: str | Path,
    cccv_rate: str = "1C",
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> tuple[List[dict], List[dict]]:
    items = discover_rpt_csvs(root)

    good = []
    bad = []

    include_pattern = (
        re.compile(include_regex, re.I)
        if include_regex else None
    )
    exclude_pattern = (
        re.compile(exclude_regex, re.I)
        if exclude_regex else None
    )

    for path, meta in items:
        cell_folder = meta.get("cell_folder", "")

        if (
            include_pattern
            and not include_pattern.search(cell_folder)
        ):
            continue

        if (
            exclude_pattern
            and exclude_pattern.search(cell_folder)
        ):
            continue

        try:
            row = extract_time_features(
                path,
                cccv_rate=cccv_rate,
                metadata=meta,
            )

            if meta.get("rpt_id_from_filename") is not None:
                row["rpt_id"] = meta["rpt_id_from_filename"]

            good.append(row)

        except Exception as e:
            bad.append(
                {
                    **meta,
                    "source_file": str(path),
                    "rpt_id": meta.get("rpt_id_from_filename"),
                    "cccv_rate_used": cccv_rate,
                    "status": "REJECTED",
                    "reason": str(e),
                }
            )

    return good, bad


def write_csv(path: str | Path, rows: List[dict]):
    path = Path(path)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract Δt, charge-time ratio, and CC/CV time features "
            "from RPT raw CSV files."
        )
    )

    ap.add_argument(
        "input",
        help=(
            "RPT Data folder, Battery raw data parent folder, "
            "or one raw RPT CSV."
        ),
    )

    ap.add_argument(
        "-o",
        "--output",
        default="time_features_all_rpt.csv",
    )

    ap.add_argument(
        "--rejected",
        default="time_features_rejected.csv",
    )

    ap.add_argument(
        "--cccv-rate",
        choices=["1C", "0.2C"],
        default="1C",
        help=(
            "Which rate supplies the canonical cc_charge_time_s and "
            "cv_charge_time_s model columns. Both rates are always "
            "saved as audit/candidate columns. Default=1C."
        ),
    )

    ap.add_argument(
        "--include-regex",
        default=None,
        help=(
            "Optional regex applied to cell folder names. "
            "Example: '25' to retain 25C folders, or "
            "'LOW C@.*25' for a narrower 25C subset."
        ),
    )

    ap.add_argument(
        "--exclude-regex",
        default=None,
        help=(
            "Optional regex applied to cell folder names after the "
            "include filter. Useful for excluding special aging "
            "conditions if their folder names are known."
        ),
    )

    args = ap.parse_args()
    root = Path(args.input)

    if root.is_file():
        good = [
            extract_time_features(
                root,
                cccv_rate=args.cccv_rate,
            )
        ]
        bad = []
    else:
        good, bad = extract_rpt_tree(
            root,
            cccv_rate=args.cccv_rate,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
        )

    write_csv(args.output, good)
    write_csv(args.rejected, bad)

    print(f"accepted RPT samples: {len(good)}")
    print(f"rejected RPT samples: {len(bad)}")
    print(f"output: {args.output}")

    if bad:
        print(f"rejected log: {args.rejected}")


if __name__ == "__main__":
    main()
