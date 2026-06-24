from __future__ import annotations

import numpy as np
import pandas as pd

def simulate_treatment_effect(
    users: pd.DataFrame,
    seed: int = 42,
    page_views_multiplier: float = 1.03,
    add_to_cart_probability_lift: float = 0.03,
    purchase_probability_lift: float = 0.02,
    revenue_multiplier: float = 1.02,
) -> pd.DataFrame:
    """Apply a realistic synthetic uplift only to treatment users."""
    rng = np.random.default_rng(seed)
    out = users.copy()
    mask = out["experiment_group"] == "treatment"

    out.loc[mask, "page_views"] = np.round(out.loc[mask, "page_views"] * page_views_multiplier).astype(int)
    out.loc[mask, "engagement_score"] = np.round(out.loc[mask, "engagement_score"] * page_views_multiplier).astype(int)

    atc_candidates = mask & (out["add_to_cart_flag"] == 0)
    atc_flip = rng.random(len(out)) < add_to_cart_probability_lift
    out.loc[atc_candidates & atc_flip, "add_to_cart_flag"] = 1
    out.loc[atc_candidates & atc_flip, "add_to_cart_count"] = 1

    purchase_candidates = mask & (out["conversion_flag"] == 0) & (out["add_to_cart_flag"] == 1)
    purchase_flip = rng.random(len(out)) < purchase_probability_lift
    converted = purchase_candidates & purchase_flip
    out.loc[converted, "conversion_flag"] = 1
    out.loc[converted, "purchase_count"] = 1
    baseline_order_value = out.loc[out["revenue"] > 0, "revenue"].median()
    if np.isnan(baseline_order_value):
        baseline_order_value = 75.0
    out.loc[converted, "revenue"] = baseline_order_value
    out.loc[mask & (out["revenue"] > 0), "revenue"] = out.loc[mask & (out["revenue"] > 0), "revenue"] * revenue_multiplier
    out["revenue_per_session"] = out["revenue"] / out["sessions"].clip(lower=1)
    return out
