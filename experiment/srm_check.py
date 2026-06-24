from __future__ import annotations

import pandas as pd
from scipy.stats import chisquare


def sample_ratio_mismatch_check(df: pd.DataFrame, group_col: str = "experiment_group", expected_ratio: float = 0.5) -> dict:
    counts = df[group_col].value_counts().reindex(["control", "treatment"]).fillna(0)
    total = counts.sum()
    expected = [total * (1 - expected_ratio), total * expected_ratio]
    stat, p_value = chisquare(counts.values, f_exp=expected)
    return {
        "control_users": int(counts["control"]),
        "treatment_users": int(counts["treatment"]),
        "chi_square_stat": float(stat),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < 0.05),
    }

