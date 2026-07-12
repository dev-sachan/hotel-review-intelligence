"""Temporal reasoning and contradiction handling.

Three products, all feeding the recommender and the dashboard:

  1. Monthly series       — per hotel (rating) and per hotel x aspect (sentiment), for the
                            trend charts.
  2. Drift detection      — is a hotel x aspect *seasonal* (Kruskal-Wallis across the four
                            seasons) or *trending* over time (Spearman rho vs date)? Every
                            test's p-value goes through a Benjamini-Hochberg FDR correction
                            because we run hundreds of tests at once, then each signal is
                            binned into an honest confidence tier (strong/moderate/weak).
  3. Contradiction index  — where do reviewers genuinely disagree, and can the disagreement
                            be *explained* by who is reviewing (verified vs not, traveler
                            type) rather than left as noise?
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config
from .taxonomy import DISPLAY_NAMES

MONTHLY_PATH = os.path.join(config.PROCESSED_DIR, "monthly.parquet")
DRIFT_PATH = os.path.join(config.PROCESSED_DIR, "drift.parquet")
CONTRA_PATH = os.path.join(config.PROCESSED_DIR, "contradictions.parquet")


# ----------------------------------------------------------------- monthly series
def build_monthly(reviews: pd.DataFrame, aspect_instances: pd.DataFrame) -> pd.DataFrame:
    """Long monthly table: rows for aspect='overall' (mean rating) and one per aspect
    (mean sentiment), keyed by (hotel_id, year_month)."""
    overall = (reviews.groupby(["hotel_id", "year_month"])
               .agg(value=("rating", "mean"), n=("rating", "size")).reset_index())
    overall["aspect"] = "overall"

    asp = aspect_instances[~aspect_instances["is_general"]]
    asp_m = (asp.groupby(["hotel_id", "aspect", "year_month"])
             .agg(value=("polarity", "mean"), n=("polarity", "size")).reset_index())

    monthly = pd.concat([overall, asp_m], ignore_index=True)
    monthly["date"] = pd.to_datetime(monthly["year_month"] + "-01")
    return monthly.sort_values(["hotel_id", "aspect", "date"]).reset_index(drop=True)


def _tier(adj_p: float) -> str:
    if np.isnan(adj_p):
        return "none"
    for thr, name in config.TIER_THRESHOLDS:
        if adj_p < thr:
            return name
    return "negligible"


# ----------------------------------------------------------------- drift
def build_drift(aspect_instances: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Seasonal (Kruskal-Wallis) + trend (Spearman) tests per hotel x aspect, plus
    aspect='overall' on ratings. Returns one row per test with BH-adjusted p and tier."""
    df = aspect_instances[~aspect_instances["is_general"]][
        ["hotel_id", "aspect", "polarity", "season", "review_date"]].copy()
    df = df.rename(columns={"polarity": "value"})

    overall = reviews[["hotel_id", "rating", "season", "review_date"]].copy()
    overall = overall.rename(columns={"rating": "value"})
    overall["aspect"] = "overall"
    df = pd.concat([df, overall], ignore_index=True)
    df["ordinal"] = pd.to_datetime(df["review_date"]).map(pd.Timestamp.toordinal)

    rows = []
    for (hotel_id, aspect), g in df.groupby(["hotel_id", "aspect"]):
        rec = {"hotel_id": hotel_id, "aspect": aspect, "n": int(len(g)),
               "kw_p": np.nan, "best_season": None, "worst_season": None,
               "season_gap": np.nan, "trend_p": np.nan, "trend_rho": np.nan,
               "trend_dir": None}

        # seasonal
        groups, means = [], {}
        for season in config.SEASON_ORDER:
            vals = g[g["season"] == season]["value"].to_numpy()
            if len(vals) >= config.MIN_SAMPLES_PER_SEASON:
                groups.append(vals)
                means[season] = float(vals.mean())
        if len(groups) >= 2:
            # Guard against the degenerate zero-variance case: when every value being
            # tested is numerically identical (which happens because the review corpus
            # is template-generated — see ASSUMPTIONS.md #1 — a hotel×aspect cell can
            # draw entirely from one template), scipy's Kruskal-Wallis divides by a
            # tie-correction factor of zero and silently returns statistic=inf,
            # pvalue=0.0 instead of raising. That reads as maximum-confidence "strong"
            # drift when in fact there is zero variation to test at all. Skip the test
            # in that case and leave kw_p as NaN (tier "none") rather than reporting a
            # spurious significant result.
            all_vals = np.concatenate(groups)
            if np.std(all_vals) > 1e-9:
                try:
                    rec["kw_p"] = float(stats.kruskal(*groups).pvalue)
                except ValueError:
                    pass
            rec["best_season"] = max(means, key=means.get)
            rec["worst_season"] = min(means, key=means.get)
            rec["season_gap"] = round(means[rec["best_season"]] - means[rec["worst_season"]], 4)

        # trend
        if len(g) >= config.MIN_SAMPLES_FOR_TREND and g["value"].nunique() > 1:
            rho, p = stats.spearmanr(g["ordinal"], g["value"])
            if not np.isnan(rho):
                rec["trend_rho"] = round(float(rho), 4)
                rec["trend_p"] = float(p)
                rec["trend_dir"] = "improving" if rho > 0 else "declining"
        rows.append(rec)

    drift = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR across all seasonal tests, and separately across all trend tests
    for pcol, tcol in [("kw_p", "seasonal_tier"), ("trend_p", "trend_tier")]:
        mask = drift[pcol].notna()
        drift[pcol.replace("_p", "_adj_p")] = np.nan
        drift[tcol] = "none"
        if mask.sum() > 0:
            adj = multipletests(drift.loc[mask, pcol], alpha=config.FDR_ALPHA,
                                method="fdr_bh")[1]
            drift.loc[mask, pcol.replace("_p", "_adj_p")] = adj
            drift.loc[mask, tcol] = [_tier(p) for p in adj]
    return drift


def seasonal_caveat(row: pd.Series) -> str | None:
    if row["seasonal_tier"] in ("none", "negligible") or row["aspect"] == "overall":
        return None
    name = DISPLAY_NAMES.get(row["aspect"], row["aspect"])
    return (f"{name} varies by season — strongest in {row['best_season']}, "
            f"weakest in {row['worst_season']} ({row['seasonal_tier']} confidence).")


def trend_caveat(row: pd.Series) -> str | None:
    if row["trend_tier"] in ("none", "negligible") or row["trend_dir"] is None:
        return None
    name = DISPLAY_NAMES.get(row["aspect"], row["aspect"]) if row["aspect"] != "overall" \
        else "Overall rating"
    return f"{name} has been {row['trend_dir']} over time ({row['trend_tier']} confidence)."


# ----------------------------------------------------------------- contradictions
def build_contradictions(aspect_instances: pd.DataFrame) -> pd.DataFrame:
    """Per hotel x aspect disagreement, with an attempt to *resolve* it by segment.

    disagreement = 2 * min(pos_share, neg_share)  in [0, 1]  (1 = perfectly split).
    When a hotel is genuinely split on an aspect, we check whether verified-vs-unverified
    or one traveler type vs the rest explains the gap, and emit a plain-language resolution.
    """
    df = aspect_instances[~aspect_instances["is_general"]].copy()
    rows = []
    for (hotel_id, aspect), g in df.groupby(["hotel_id", "aspect"]):
        w = g["weight"].to_numpy()
        s = g["polarity"].to_numpy()
        pos = float(w[s >= config.POLARITY_THRESHOLD].sum())
        neg = float(w[s <= -config.POLARITY_THRESHOLD].sum())
        tot = pos + neg
        if tot <= 0 or len(g) < 10:
            continue
        pos_share = pos / tot
        disagreement = round(2 * min(pos_share, 1 - pos_share), 4)
        if disagreement < 0.40:
            continue

        resolution = _resolve_segment(g)
        rows.append({
            "hotel_id": hotel_id, "aspect": aspect, "n": int(len(g)),
            "pos_share": round(pos_share, 3), "disagreement": disagreement,
            "resolution": resolution,
        })
    return pd.DataFrame(rows)


def _seg_mean(g: pd.DataFrame) -> float:
    w = g["weight"].to_numpy()
    return float((g["polarity"].to_numpy() * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def _resolve_segment(g: pd.DataFrame) -> str | None:
    """Try verified split, then traveler-type split; return the most explanatory story."""
    best = None
    best_gap = 0.35  # only report a split that explains a meaningful gap

    # verified vs unverified
    ver, unver = g[g["verified"]], g[~g["verified"]]
    if len(ver) >= 5 and len(unver) >= 5:
        gap = _seg_mean(ver) - _seg_mean(unver)
        if abs(gap) >= best_gap:
            best_gap = abs(gap)
            lean = "more positive" if gap > 0 else "more negative"
            best = f"Verified reviewers are {lean} ({_seg_mean(ver):+.2f} vs {_seg_mean(unver):+.2f} unverified)."

    # one traveler type vs the rest
    for tt in ["business", "family", "couple", "solo", "group", "leisure"]:
        seg = g[g["traveler_type"] == tt]
        rest = g[g["traveler_type"] != tt]
        if len(seg) >= 5 and len(rest) >= 5:
            gap = _seg_mean(seg) - _seg_mean(rest)
            if abs(gap) >= best_gap:
                best_gap = abs(gap)
                lean = "rate it higher" if gap > 0 else "rate it lower"
                best = f"{tt.capitalize()} travelers {lean} ({_seg_mean(seg):+.2f} vs {_seg_mean(rest):+.2f} for others)."
    return best