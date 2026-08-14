from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_IDS = ["T0_FIXED_RAW_PM1PCT", "T1_VOLATILITY_SCALED_RAW", "T2_MARKET_RELATIVE_FIXED"]


def _ordinal(values: pd.Series, threshold: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.notna() & threshold.notna()
    result.loc[valid & (values < -threshold)] = 0
    result.loc[valid & values.between(-threshold, threshold, inclusive="both")] = 1
    result.loc[valid & (values > threshold)] = 2
    return result


def build_target_variants(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"missing target audit columns: {missing}")
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise").dt.normalize()
    raw = pd.to_numeric(result["target_return_h4"], errors="coerce")
    base_valid = result["target_valid"].astype(bool) & raw.notna()
    market = raw.where(base_valid).groupby(result["trade_date"], sort=True).transform("median")
    volatility = pd.to_numeric(result["realized_volatility_8w"], errors="coerce")
    result["market_h4_median"] = market
    result["T0_return"] = raw
    result["T0_threshold"] = 0.01
    result["T0_valid"] = base_valid
    result["T0_label"] = _ordinal(raw.where(base_valid), pd.Series(0.01, index=result.index))
    result["T1_return"] = raw
    result["T1_threshold"] = (0.25 * volatility).clip(lower=0.005, upper=0.03)
    result["T1_valid"] = base_valid & result["T1_threshold"].notna()
    result["T1_label"] = _ordinal(raw.where(result["T1_valid"]), result["T1_threshold"])
    result["T2_return"] = raw - market
    result["T2_threshold"] = 0.01
    result["T2_valid"] = base_valid & market.notna()
    result["T2_label"] = _ordinal(
        result["T2_return"].where(result["T2_valid"]), pd.Series(0.01, index=result.index)
    )
    return result


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    middle = 0.5 * (p + q)
    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))
    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def audit_target(frame: pd.DataFrame, prefix: str) -> tuple[dict, pd.DataFrame]:
    valid = frame[f"{prefix}_valid"].astype(bool)
    eligible = frame.loc[valid].copy()
    label = eligible[f"{prefix}_label"].astype(int)
    counts = np.bincount(label, minlength=3).astype(float)
    global_share = counts / counts.sum()
    weekly_rows = []
    for trade_date, group in eligible.groupby("trade_date", sort=True):
        weekly_count = np.bincount(group[f"{prefix}_label"].astype(int), minlength=3).astype(float)
        weekly_share = weekly_count / weekly_count.sum()
        weekly_rows.append({
            "trade_date": trade_date, "eligible_rows": len(group),
            "down_share": weekly_share[0], "neutral_share": weekly_share[1], "up_share": weekly_share[2],
            "direction_share": weekly_share[2] - weekly_share[0],
            "market_h4_median": float(group["market_h4_median"].iloc[0]),
            "js_from_global": _js_divergence(weekly_share, global_share),
        })
    weekly = pd.DataFrame(weekly_rows)
    dependence = weekly["direction_share"].rank(method="average").corr(
        weekly["market_h4_median"].rank(method="average")
    )
    distance = (eligible[f"{prefix}_return"].abs() - eligible[f"{prefix}_threshold"]).abs()
    ambiguity = (distance <= 0.25 * eligible[f"{prefix}_threshold"]).mean()
    summary = {
        "target_id": {
            "T0": "T0_FIXED_RAW_PM1PCT", "T1": "T1_VOLATILITY_SCALED_RAW",
            "T2": "T2_MARKET_RELATIVE_FIXED",
        }[prefix],
        "eligible_rows": int(len(eligible)),
        "eligible_weeks": int(len(weekly)),
        "class_counts": {"down": int(counts[0]), "neutral": int(counts[1]), "up": int(counts[2])},
        "class_shares": {"down": float(global_share[0]), "neutral": float(global_share[1]), "up": float(global_share[2])},
        "minimum_global_class_share": float(global_share.min()),
        "median_weekly_eligible_rows": float(weekly["eligible_rows"].median()),
        "weekly_distribution_instability_js_mean": float(weekly["js_from_global"].mean()),
        "market_regime_dependence_abs_spearman": float(abs(dependence)),
        "boundary_ambiguity_fraction": float(ambiguity),
    }
    return summary, weekly


def select_target(summaries: list[dict]) -> str:
    reference = next(item for item in summaries if item["target_id"] == "T0_FIXED_RAW_PM1PCT")
    candidates = []
    for item in summaries:
        if item["target_id"] == "T0_FIXED_RAW_PM1PCT":
            continue
        coverage = item["eligible_rows"] / reference["eligible_rows"]
        item["coverage_vs_reference"] = float(coverage)
        item["admissible"] = bool(
            coverage >= 0.99 and item["minimum_global_class_share"] >= 0.08
            and item["median_weekly_eligible_rows"] >= 300
        )
        if item["admissible"]:
            candidates.append(item)
    if not candidates:
        raise RuntimeError("no admissible REV8 target candidate")
    selected = min(candidates, key=lambda item: (
        item["market_regime_dependence_abs_spearman"],
        item["weekly_distribution_instability_js_mean"],
        item["boundary_ambiguity_fraction"], item["target_id"],
    ))
    return selected["target_id"]


