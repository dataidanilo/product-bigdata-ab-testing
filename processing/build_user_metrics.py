from __future__ import annotations

import pandas as pd


def build_user_metrics(sessions: pd.DataFrame) -> pd.DataFrame:
    users = sessions.groupby("user_pseudo_id", as_index=False).agg(
        sessions=("ga_session_id", "nunique"),
        page_views=("page_views", "sum"),
        view_items=("view_items", "sum"),
        add_to_cart_count=("add_to_cart_count", "sum"),
        purchase_count=("purchase_count", "sum"),
        revenue=("revenue", "sum"),
        avg_session_duration_seconds=("session_duration_seconds", "mean"),
        engagement_score=("engagement_score", "sum"),
        country=("country", "first"),
        source=("source", "first"),
    )
    users["add_to_cart_flag"] = (users["add_to_cart_count"] > 0).astype(int)
    users["conversion_flag"] = (users["purchase_count"] > 0).astype(int)
    users["revenue_per_session"] = users["revenue"] / users["sessions"].clip(lower=1)
    return users