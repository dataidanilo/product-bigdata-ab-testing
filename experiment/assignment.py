from __future__ import annotations

import hashlib
import pandas as pd



def _hash_to_unit_interval(value: str, seed: int = 42) -> float:
    digest = hashlib.md5(f"{value}_{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF



def assign_users(users: pd.DataFrame, treatment_ratio: float = 0.5, seed: int = 42) -> pd.DataFrame:
    if not 0 < treatment_ratio < 1:
        raise ValueError("treatment_ratio must be between 0 and 1")
    out = users.copy()
    scores = out["user_pseudo_id"].astype(str).map(lambda x: _hash_to_unit_interval(x, seed))
    out["experiment_group"] = scores.map(lambda x: "treatment" if x < treatment_ratio else "control")
    return out

