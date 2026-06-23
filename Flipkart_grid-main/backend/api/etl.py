"""
backend/etl.py — Robust CSV ingestion pipeline (Pillar 5).

Steps:
  1. Schema check with fuzzy header mapping (rapidfuzz if available, else difflib).
  2. Auto-cleaning: timestamp parse, Bengaluru boundary check, dtype downcast.
  3. Derived features + CIS using the existing helpers (single source of truth).

Pure-Python (pandas/numpy only) so it runs anywhere; no cloud deps.
"""
from __future__ import annotations

import io
import io
from difflib import get_close_matches
from typing import Optional

import numpy as np
import pandas as pd

from api.helpers import (
    get_max_severity,
    get_vehicle_size,
    parse_violations,
)

# Bengaluru metro bounding box.
LAT_MIN, LAT_MAX = 12.8, 13.1
LON_MIN, LON_MAX = 77.4, 77.8

REQUIRED = ["latitude", "longitude", "created_datetime", "police_station"]
CANONICAL = REQUIRED + ["vehicle_type", "violation_type", "junction_name", "location", "id"]

# Common header aliases → canonical name (checked before fuzzy matching).
ALIASES = {
    "lat": "latitude",
    "lng": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "timestamp": "created_datetime",
    "datetime": "created_datetime",
    "date_time": "created_datetime",
    "date": "created_datetime",
    "station": "police_station",
    "ps": "police_station",
    "vehicle": "vehicle_type",
    "offense": "violation_type",
    "violation": "violation_type",
    "junction": "junction_name",
    "place": "location",
    "address": "location",
}

try:
    from rapidfuzz import process as _rf_process  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:  # noqa: BLE001
    _HAS_RAPIDFUZZ = False


def _fuzzy_match(col: str, choices: list[str]) -> Optional[str]:
    if _HAS_RAPIDFUZZ:
        m = _rf_process.extractOne(col, choices, score_cutoff=72)
        return m[0] if m else None
    m = get_close_matches(col, choices, n=1, cutoff=0.72)
    return m[0] if m else None


def map_columns(columns: list[str]) -> tuple[dict, list[str]]:
    """Return (rename_map, unmapped_canonical). rename_map: original → canonical."""
    rename: dict[str, str] = {}
    taken: set[str] = set()
    for col in columns:
        key = str(col).strip().lower().replace(" ", "_")
        target = None
        if key in CANONICAL:
            target = key
        elif key in ALIASES:
            target = ALIASES[key]
        else:
            target = _fuzzy_match(key, [c for c in CANONICAL if c not in taken])
        if target and target not in taken:
            rename[col] = target
            taken.add(target)
    missing = [c for c in REQUIRED if c not in taken]
    return rename, missing


def clean_dataframe(source, column_map: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Ingest a CSV (path | bytes | DataFrame) → (clean_df, stats).

    `column_map` (optional) is a manual override {canonical_field: source_column}
    supplied by the user when auto fuzzy-mapping is ambiguous.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(source), low_memory=False)
    else:
        df = pd.read_csv(source, low_memory=False)

    raw_rows = len(df)
    rename, missing = map_columns(list(df.columns))

    # Apply manual overrides (win over auto-mapping for those targets).
    if column_map:
        manual_targets = {t for t, s in column_map.items() if s}
        rename = {src: tgt for src, tgt in rename.items() if tgt not in manual_targets}
        for target, src in column_map.items():
            if src and src in df.columns:
                rename[src] = target
        mapped = set(rename.values())
        missing = [c for c in REQUIRED if c not in mapped]

    df = df.rename(columns=rename)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # ── timestamps ──
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    dropped_dt = int(df["created_datetime"].isna().sum())
    df = df.dropna(subset=["created_datetime"])

    # ── coordinates + Bengaluru boundary check ──
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    before_geo = len(df)
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) & df["longitude"].between(LON_MIN, LON_MAX)
    ]
    dropped_coords = before_geo - len(df)

    # ── derived features ──
    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.day_name()
    df["day_num"] = df["created_datetime"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7 <= h <= 10 or 16 <= h <= 20) else 0)
    df["is_weekend"] = df["day_num"].apply(lambda d: 1 if d >= 5 else 0)

    if "violation_type" in df.columns:
        df["violation_list"] = df["violation_type"].apply(parse_violations)
        df["violation_severity"] = df["violation_list"].apply(get_max_severity)
    else:
        df["violation_list"] = [[] for _ in range(len(df))]
        df["violation_severity"] = 3
    df["vehicle_size_score"] = (
        df["vehicle_type"].apply(get_vehicle_size) if "vehicle_type" in df.columns else 3
    )
    if "junction_name" in df.columns:
        df["junction_factor"] = df["junction_name"].apply(
            lambda j: 1.0 if pd.isna(j) or str(j).strip().upper() == "NO JUNCTION" else 2.0
        )
    else:
        df["junction_factor"] = 1.0

    # Vectorized time factor (same logic as compute_time_factor) — fast on 300k+ rows.
    hour, wknd = df["hour"], df["is_weekend"]
    df["time_factor"] = (
        np.select(
            [hour.isin([8, 9, 10, 17, 18, 19]), hour.isin([7, 11, 16, 20]), hour.between(11, 16), hour.between(21, 23)],
            [3.0, 2.0, 1.5, 1.0],
            default=0.5,
        )
        * np.where(wknd == 1, 0.7, 1.0)
    )

    # ── CIS (0–100) ──
    raw_cis = df["violation_severity"] * df["vehicle_size_score"] * df["time_factor"] * df["junction_factor"]
    cmin, cmax = raw_cis.min(), raw_cis.max()
    df["cis"] = ((raw_cis - cmin) / (cmax - cmin) * 100) if cmax > cmin else 50.0

    # ── dtype downcast (memory) ──
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    stats = {
        "raw_rows": raw_rows,
        "final_rows": len(df),
        "dropped_datetime": dropped_dt,
        "dropped_coords": dropped_coords,
        "mapped_columns": rename,
        "date_range": (
            f"{df['created_datetime'].min():%d %b %Y} – {df['created_datetime'].max():%d %b %Y}"
            if len(df) else "—"
        ),
        "stations": int(df["police_station"].nunique()) if len(df) else 0,
        "locations": int(df["location"].nunique()) if "location" in df.columns and len(df) else 0,
        "mem_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    return df, stats


if __name__ == "__main__":  # quick self-test
    demo = pd.DataFrame(
        {
            "Lat": [12.93, 99.0, 12.97],
            "Lng": [77.62, 77.62, 77.61],
            "Timestamp": ["2024-01-01 09:00", "2024-01-02 18:00", "bad"],
            "Station": ["Koramangala", "Silk Board", "MG Road"],
            "Vehicle": ["Car", "HGV", "Bus"],
        }
    )
    out, st = clean_dataframe(demo)
    print("rows:", st["raw_rows"], "->", st["final_rows"], "| dropped_dt:", st["dropped_datetime"], "| dropped_coords:", st["dropped_coords"])
    print("mapped:", st["mapped_columns"])
