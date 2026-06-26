from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


def summarize_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.groupby("experiment_group")[metric].agg(["count", "mean", "std", "sum"]).reset_index()


def ab_test_continuous(df: pd.DataFrame, metric: str) -> dict:
    control = df.loc[df["experiment_group"] == "control", metric].astype(float)
    treatment = df.loc[df["experiment_group"] == "treatment", metric].astype(float)
    stat, p_value = stats.ttest_ind(treatment, control, equal_var=False, nan_policy="omit")
    diff = treatment.mean() - control.mean()
    uplift = diff / control.mean() if control.mean() != 0 else np.nan
    return {
        "metric": metric,
        "control_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "absolute_difference": float(diff),
        "relative_uplift": float(uplift),
        "p_value": float(p_value),
        "test": "welch_t_test",
    }


def ab_test_binary(df: pd.DataFrame, metric: str) -> dict:
    control = df[df["experiment_group"] == "control"][metric].astype(int)
    treatment = df[df["experiment_group"] == "treatment"][metric].astype(int)
    count = np.array([treatment.sum(), control.sum()])
    nobs = np.array([len(treatment), len(control)])
    stat, p_value = proportions_ztest(count, nobs)
    diff = treatment.mean() - control.mean()
    uplift = diff / control.mean() if control.mean() != 0 else np.nan
    ci_low, ci_high = confint_proportions_2indep(count1=int(treatment.sum()), nobs1=len(treatment), count2=int(control.sum()), nobs2=len(control), method="wald")
    return {
        "metric": metric,
        "control_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "absolute_difference": float(diff),
        "relative_uplift": float(uplift),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "test": "two_proportion_z_test",
    }


def run_experiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    binary_metrics = ["conversion_flag", "add_to_cart_flag"]
    continuous_metrics = ["revenue", "revenue_per_session", "engagement_score", "page_views"]
    results = []
    for metric in binary_metrics:
        if metric in df.columns:
            results.append(ab_test_binary(df, metric))
    for metric in continuous_metrics:
        if metric in df.columns:
            results.append(ab_test_continuous(df, metric))
    return pd.DataFrame(results)
