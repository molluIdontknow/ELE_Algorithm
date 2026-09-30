#!/usr/bin/env python3
"""
dcir_feature_extractor_v2.py

DCIR extractor using an 8–10 s WINDOW instead of one fixed time point.

Default project definition
--------------------------
For each pulse:
    R(t) = |V(t) - V(0)| / |I(t) - I(0)|

for all available samples in 8 <= t <= 10 s.

The representative training feature is:
    dcir_project_ohm = mean(R(t), 8–10 s)

The script also stores median/std/min/max and the exact start/end-point DCIR
for QC. Therefore you do not lose the 8 s / 10 s values.

For SOC 0,10,...,80%, the same 8–10 s window method is used.

If the real hardware later uses a different window, change:
    --window-start
    --window-end

Example:
    --window-start 8 --window-end 10

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


NOMINAL_CAPACITY_AH = 3.0
TARGET_GITT_A = 0.2 * NOMINAL_CAPACITY_AH   # 0.6 A
TARGET_DCIR_A = 3.0

CURRENT_REL_TOL = 0.12
CURRENT_ABS_TOL_MIN = 0.05

PREFERRED_GITT_STEP = "20"
PREFERRED_3A_CHARGE_STEP = "37"
PREFERRED_3A_DISCHARGE_STEP = "39"
REFERENCE_REPEAT = "1"


class DCIRFeatureError(RuntimeError):
    pass


def _f(value) -> float:
    try:
        s = str(value).strip()
        return float(s) if s else math.nan
    except Exception:
        return math.nan


def parse_hms_seconds(value) -> float:
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
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {
        "StepNo",
        "Type",
        "CurCycle",
        "TotCycle",
        "StepTime(H:M:S)",
        "Voltage(V)",
        "Current(A)",
    }

    missing = required.difference(fields)
    if missing:
        raise DCIRFeatureError(
            f"Missing required columns: {sorted(missing)}"
        )

    return rows


def contiguous_segments(rows: List[dict]) -> List[Tuple[tuple, List[dict]]]:
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
        raise DCIRFeatureError(
            "Too few valid time/voltage/current points."
        )

    order = np.argsort(t, kind="stable")
    return t[order], v[order], i[order]


def _collapse_duplicate_times(t, v, i):
    unique_t = np.unique(t)

    vv = np.array(
        [v[t == x].mean() for x in unique_t],
        dtype=float,
    )
    ii = np.array(
        [i[t == x].mean() for x in unique_t],
        dtype=float,
    )

    return unique_t, vv, ii


def interpolate_vi(rows: List[dict], seconds: float) -> Tuple[float, float]:
    t, v, i = segment_arrays(rows)
    t, v, i = _collapse_duplicate_times(t, v, i)

    if seconds < t.min() - 1e-9 or seconds > t.max() + 1e-9:
        raise DCIRFeatureError(
            f"Requested {seconds:g} s but segment covers "
            f"{t.min():g}–{t.max():g} s."
        )

    return (
        float(np.interp(seconds, t, v)),
        float(np.interp(seconds, t, i)),
    )


def window_sample_times(
    rows: List[dict],
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """
    Use every raw sample in the requested interval and force exact start/end
    times into the set by interpolation if necessary.
    """
    if end_s < start_s:
        raise DCIRFeatureError(
            "window_end must be >= window_start."
        )

    t, _, _ = segment_arrays(rows)

    if start_s < np.min(t) - 1e-9 or end_s > np.max(t) + 1e-9:
        raise DCIRFeatureError(
            f"Requested window {start_s:g}–{end_s:g} s "
            f"but segment covers {np.min(t):g}–{np.max(t):g} s."
        )

    inside = t[(t >= start_s) & (t <= end_s)]

    times = np.unique(
        np.concatenate(
            [
                inside,
                np.array([start_s, end_s], dtype=float),
            ]
        )
    )

    return np.sort(times)


def dcir_series_from_rest(
    rows: List[dict],
    start_s: float,
    end_s: float,
):
    """
    R(t) = |V(t)-V(0)| / |I(t)-I(0)| over the requested window.
    """
    v0, i0 = interpolate_vi(rows, 0.0)
    times = window_sample_times(rows, start_s, end_s)

    values = []

    for sec in times:
        vt, it = interpolate_vi(rows, float(sec))
        di = abs(it - i0)

        if di < 1e-9:
            continue

        r = abs(vt - v0) / di
        values.append((float(sec), float(r), vt, it))

    if not values:
        raise DCIRFeatureError(
            "No valid DCIR values in requested time window."
        )

    return values


def summarize_series(values):
    r = np.array([x[1] for x in values], dtype=float)

    return {
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "std": float(np.std(r, ddof=1)) if len(r) >= 2 else 0.0,
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "n": int(len(r)),
        "start": float(values[0][1]),
        "end": float(values[-1][1]),
    }


def median_active_current(rows: List[dict]) -> float:
    _, _, i = segment_arrays(rows)
    active = i[np.abs(i) > 0.05]

    return (
        float(np.median(active))
        if len(active)
        else math.nan
    )


def duration_s(rows: List[dict]) -> float:
    t, _, _ = segment_arrays(rows)
    return float(np.max(t) - np.min(t))


def find_pulse_segment(
    rows: List[dict],
    direction: str,
    current_a: float,
    repeat: str,
    preferred_step: str | None,
    window_end_s: float,
):
    target_sign = 1 if direction == "charge" else -1
    candidates = []

    for key, seg_rows in contiguous_segments(rows):
        step_no, typ, curcycle, totcycle = key

        if typ != direction:
            continue

        if str(curcycle) != str(repeat):
            continue

        imed = median_active_current(seg_rows)
        dur = duration_s(seg_rows)

        if not math.isfinite(imed):
            continue

        tol = max(
            CURRENT_ABS_TOL_MIN,
            abs(current_a) * CURRENT_REL_TOL,
        )

        if abs(imed - target_sign * abs(current_a)) > tol:
            continue

        if dur + 1e-9 < window_end_s or dur > 60.0:
            continue

        candidates.append(
            {
                "key": key,
                "rows": seg_rows,
                "imed": imed,
                "duration_s": dur,
            }
        )

    if not candidates:
        raise DCIRFeatureError(
            f"No {direction} pulse near {current_a:g} A "
            f"covering {window_end_s:g} s."
        )

    if preferred_step is not None:
        exact = [
            c for c in candidates
            if c["key"][0] == str(preferred_step)
        ]
        if exact:
            return exact[-1]

    return candidates[-1]


def paired_dcir_series(
    charge_rows: List[dict],
    discharge_rows: List[dict],
    start_s: float,
    end_s: float,
):
    """
    R_pair(t) =
        |V_charge(t)-V_discharge(t)| /
        |I_charge(t)-I_discharge(t)|
    """
    tc = window_sample_times(charge_rows, start_s, end_s)
    td = window_sample_times(discharge_rows, start_s, end_s)

    times = np.unique(
        np.concatenate(
            [
                tc,
                td,
                np.array([start_s, end_s], dtype=float),
            ]
        )
    )

    values = []

    for sec in np.sort(times):
        vch, ich = interpolate_vi(charge_rows, float(sec))
        vdis, idis = interpolate_vi(discharge_rows, float(sec))

        di = abs(ich - idis)

        if di < 1e-9:
            continue

        r = abs(vch - vdis) / di
        values.append((float(sec), float(r), math.nan, math.nan))

    if not values:
        raise DCIRFeatureError(
            "No valid paired DCIR values in requested window."
        )

    return values


def extract_project_3a_soc50(
    rows: List[dict],
    start_s: float,
    end_s: float,
    repeat: str = REFERENCE_REPEAT,
):
    ch = find_pulse_segment(
        rows,
        "charge",
        TARGET_DCIR_A,
        repeat,
        PREFERRED_3A_CHARGE_STEP,
        end_s,
    )
    dis = find_pulse_segment(
        rows,
        "discharge",
        TARGET_DCIR_A,
        repeat,
        PREFERRED_3A_DISCHARGE_STEP,
        end_s,
    )

    ch_values = dcir_series_from_rest(
        ch["rows"],
        start_s,
        end_s,
    )
    dis_values = dcir_series_from_rest(
        dis["rows"],
        start_s,
        end_s,
    )
    pair_values = paired_dcir_series(
        ch["rows"],
        dis["rows"],
        start_s,
        end_s,
    )

    ch_s = summarize_series(ch_values)
    dis_s = summarize_series(dis_values)
    pair_s = summarize_series(pair_values)

    return {
        "charge": ch_s,
        "discharge": dis_s,
        "pair": pair_s,
        "charge_step": ch["key"][0],
        "discharge_step": dis["key"][0],
        "repeat": repeat,
    }


def find_gitt_charge_segments(
    rows: List[dict],
    window_end_s: float,
) -> Dict[int, dict]:
    """
    Supplied RPT structure:
      CurCycle 1 -> SOC 0
      CurCycle 2 -> SOC 10
      ...
      CurCycle 9 -> SOC 80
    """
    found = {}

    for key, seg_rows in contiguous_segments(rows):
        step_no, typ, curcycle, totcycle = key

        if typ != "charge":
            continue

        try:
            cyc = int(float(curcycle))
        except Exception:
            continue

        if cyc < 1 or cyc > 9:
            continue

        imed = median_active_current(seg_rows)
        dur = duration_s(seg_rows)

        tol = max(
            CURRENT_ABS_TOL_MIN,
            TARGET_GITT_A * CURRENT_REL_TOL,
        )

        if (
            not math.isfinite(imed)
            or abs(imed - TARGET_GITT_A) > tol
            or dur + 1e-9 < window_end_s
        ):
            continue

        score = (
            1 if step_no == PREFERRED_GITT_STEP
            else 0
        )

        old = found.get(cyc)

        if old is None or score > old["score"]:
            found[cyc] = {
                "score": score,
                "key": key,
                "rows": seg_rows,
            }

    return found


def extract_soc0_80_window(
    rows: List[dict],
    start_s: float,
    end_s: float,
):
    segs = find_gitt_charge_segments(
        rows,
        end_s,
    )

    result = {}

    for soc in range(0, 81, 10):
        cyc = soc // 10 + 1
        seg = segs.get(cyc)

        prefix = f"dcir_soc{soc:02d}"

        if seg is None:
            result[f"{prefix}_window_mean_ohm"] = math.nan
            result[f"{prefix}_window_median_ohm"] = math.nan
            result[f"{prefix}_window_std_ohm"] = math.nan
            result[f"{prefix}_window_start_ohm"] = math.nan
            result[f"{prefix}_window_end_ohm"] = math.nan
            continue

        vals = dcir_series_from_rest(
            seg["rows"],
            start_s,
            end_s,
        )
        s = summarize_series(vals)

        result[f"{prefix}_window_mean_ohm"] = s["mean"]
        result[f"{prefix}_window_median_ohm"] = s["median"]
        result[f"{prefix}_window_std_ohm"] = s["std"]
        result[f"{prefix}_window_start_ohm"] = s["start"]
        result[f"{prefix}_window_end_ohm"] = s["end"]

    return result


def parse_channel_number(channel: str) -> int | None:
    m = re.search(
        r"Ch\s*0*(\d+)",
        str(channel),
        flags=re.I,
    )

    return int(m.group(1)) if m else None


def parse_cell_folder(folder_name: str) -> dict:
    raw = str(folder_name).strip()

    m = re.match(
        r"^\s*(\d+)\.\s*(.+?)\s*$",
        raw,
    )

    global_id = (
        int(m.group(1))
        if m
        else None
    )

    label = (
        m.group(2).strip()
        if m
        else raw
    )

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
    m = re.fullmatch(
        r"\s*(\d+)\s*",
        path.stem,
    )

    return int(m.group(1)) if m else None


def extract_dcir_features(
    path: str | Path,
    window_start_s: float = 8.0,
    window_end_s: float = 10.0,
    project_mode: str = "discharge",
    metadata: dict | None = None,
):
    path = Path(path)
    rows = read_raw_csv(path)

    if window_start_s < 0:
        raise DCIRFeatureError(
            "window_start must be >= 0."
        )

    if window_end_s < window_start_s:
        raise DCIRFeatureError(
            "window_end must be >= window_start."
        )

    project = extract_project_3a_soc50(
        rows,
        window_start_s,
        window_end_s,
    )

    if project_mode not in {
        "charge",
        "discharge",
        "pair",
    }:
        raise DCIRFeatureError(
            "project_mode must be charge, discharge, or pair."
        )

    selected = project[project_mode]

    channel = (
        str(rows[0].get("Channel", "")).strip()
        if rows else ""
    )
    channel_num = parse_channel_number(channel)

    out = {
        "source_file": str(path),
        "channel": channel,
        "channel_num": channel_num,
        "rpt_id": parse_rpt_id(path),

        "window_start_s": float(window_start_s),
        "window_end_s": float(window_end_s),
        "project_mode": project_mode,

        # Main model DCIR feature.
        "dcir_project_ohm": selected["mean"],

        # QC / candidate window statistics.
        "dcir_project_window_mean_ohm": selected["mean"],
        "dcir_project_window_median_ohm": selected["median"],
        "dcir_project_window_std_ohm": selected["std"],
        "dcir_project_window_min_ohm": selected["min"],
        "dcir_project_window_max_ohm": selected["max"],
        "dcir_project_window_start_ohm": selected["start"],
        "dcir_project_window_end_ohm": selected["end"],
        "dcir_project_window_n": selected["n"],

        # Keep all three SOC50 3-A definitions for audit/comparison.
        "dcir_soc50_3a_charge_window_mean_ohm":
            project["charge"]["mean"],
        "dcir_soc50_3a_discharge_window_mean_ohm":
            project["discharge"]["mean"],
        "dcir_soc50_3a_pair_window_mean_ohm":
            project["pair"]["mean"],

        "soc50_3a_charge_step":
            project["charge_step"],
        "soc50_3a_discharge_step":
            project["discharge_step"],
        "soc50_3a_repeat":
            project["repeat"],

        **extract_soc0_80_window(
            rows,
            window_start_s,
            window_end_s,
        ),

        "status": "OK",
    }

    if metadata:
        out = {
            **metadata,
            **out,
        }

    folder_id = out.get(
        "cell_global_id"
    )

    if (
        folder_id is not None
        and channel_num is not None
    ):
        out["cell_channel_match"] = bool(
            int(folder_id) == int(channel_num)
        )
    else:
        out["cell_channel_match"] = None

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

    raise DCIRFeatureError(
        "Input directory must be RPT Data or its "
        "Battery raw data parent folder."
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
        meta = parse_cell_folder(
            cell_dir.name
        )

        for f in cell_dir.iterdir():
            if (
                not f.is_file()
                or f.suffix.lower()
                not in {".csv", ".txt"}
            ):
                continue

            rpt_id = parse_rpt_id(f)

            if rpt_id is None:
                continue

            found.append(
                (
                    f,
                    {
                        **meta,
                        "rpt_id_from_filename":
                            rpt_id,
                    },
                )
            )

    found.sort(
        key=lambda item: (
            item[1].get(
                "cell_global_id"
            )
            if item[1].get(
                "cell_global_id"
            ) is not None
            else 10**9,
            item[1].get(
                "rpt_id_from_filename",
                10**9,
            ),
            str(item[0]),
        )
    )

    return found


def extract_rpt_tree(
    root: str | Path,
    window_start_s: float = 8.0,
    window_end_s: float = 10.0,
    project_mode: str = "discharge",
    include_regex: str | None = None,
):
    items = discover_rpt_csvs(root)

    good = []
    bad = []

    pattern = (
        re.compile(
            include_regex,
            re.I,
        )
        if include_regex
        else None
    )

    for path, meta in items:
        if (
            pattern
            and not pattern.search(
                meta.get(
                    "cell_folder",
                    "",
                )
            )
        ):
            continue

        try:
            row = extract_dcir_features(
                path,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                project_mode=project_mode,
                metadata=meta,
            )

            if (
                meta.get(
                    "rpt_id_from_filename"
                )
                is not None
            ):
                row["rpt_id"] = meta[
                    "rpt_id_from_filename"
                ]

            good.append(row)

        except Exception as e:
            bad.append(
                {
                    **meta,
                    "source_file": str(path),
                    "rpt_id":
                        meta.get(
                            "rpt_id_from_filename"
                        ),
                    "window_start_s":
                        window_start_s,
                    "window_end_s":
                        window_end_s,
                    "project_mode":
                        project_mode,
                    "status": "REJECTED",
                    "reason": str(e),
                }
            )

    return good, bad


def write_csv(
    path: str | Path,
    rows: List[dict],
):
    path = Path(path)

    if not rows:
        path.write_text(
            "",
            encoding="utf-8-sig",
        )
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
            "Extract DCIR from an 8–10 s "
            "measurement window."
        )
    )

    ap.add_argument(
        "input",
        help=(
            "RPT Data folder, Battery raw data "
            "folder, or one raw RPT CSV."
        ),
    )

    ap.add_argument(
        "-o",
        "--output",
        default="dcir_features_window.csv",
    )

    ap.add_argument(
        "--rejected",
        default="dcir_features_window_rejected.csv",
    )

    ap.add_argument(
        "--window-start",
        type=float,
        default=8.0,
    )

    ap.add_argument(
        "--window-end",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--project-mode",
        choices=[
            "discharge",
            "charge",
            "pair",
        ],
        default="discharge",
    )

    ap.add_argument(
        "--include-regex",
        default=None,
        help=(
            "Example: '25' for 25C aging folders, "
            "or 'LOW C@.*25' for a narrower subset."
        ),
    )

    args = ap.parse_args()
    root = Path(args.input)

    if root.is_file():
        good = [
            extract_dcir_features(
                root,
                window_start_s=
                    args.window_start,
                window_end_s=
                    args.window_end,
                project_mode=
                    args.project_mode,
            )
        ]
        bad = []
    else:
        good, bad = extract_rpt_tree(
            root,
            window_start_s=
                args.window_start,
            window_end_s=
                args.window_end,
            project_mode=
                args.project_mode,
            include_regex=
                args.include_regex,
        )

    write_csv(
        args.output,
        good,
    )
    write_csv(
        args.rejected,
        bad,
    )

    print(
        f"accepted RPT samples: "
        f"{len(good)}"
    )
    print(
        f"rejected RPT samples: "
        f"{len(bad)}"
    )
    print(
        f"DCIR window: "
        f"{args.window_start:g}–"
        f"{args.window_end:g} s"
    )
    print(
        f"project mode: "
        f"{args.project_mode}"
    )
    print(
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
