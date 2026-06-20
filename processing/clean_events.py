from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "user_pseudo_id",
    "ga_session_id",
    "event_name",
    "event_timestamp",
    "event_date",
    "device_category",
    "country",
    "source",
    "revenue",
]


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out = out.dropna(subset=["user_pseudo_id", "ga_session_id", "event_name"])
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce").fillna(0.0)
    out["event_timestamp"] = pd.to_numeric(out["event_timestamp"], errors="coerce")
    out = out.dropna(subset=["event_timestamp"])
    out["event_time"] = pd.to_datetime(out["event_timestamp"], unit="us")
    return out