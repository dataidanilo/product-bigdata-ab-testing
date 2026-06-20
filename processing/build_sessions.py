from __future__ import annotations

import pandas as pd


def build_session_metrics(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(["user_pseudo_id", "ga_session_id"], as_index=False)
    sessions = grouped.agg(
        session_start=("event_time", "min"),
        session_end=("event_time", "max"),
        page_views=("event_name", lambda x: int((x == "page_view").sum())),
        view_items=("event_name", lambda x: int((x == "view_item").sum())),
        add_to_cart_count=("event_name", lambda x: int((x == "add_to_cart").sum())),
        purchase_count=("event_name", lambda x: int((x == "purchase").sum())),
        revenue=("revenue", "sum"),
        device_category=("device_category", "first"),
        country=("country", "first"),
        source=("source", "first"),
    )
    sessions["session_duration_seconds"] = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().clip(lower=0)
    sessions["add_to_cart_flag"] = (sessions["add_to_cart_count"] > 0).astype(int)
    sessions["conversion_flag"] = (sessions["purchase_count"] > 0).astype(int)
    sessions["engagement_score"] = sessions["page_views"] + 2 * sessions["view_items"] + 4 * sessions["add_to_cart_count"] + 8 * sessions["purchase_count"]
    return sessions