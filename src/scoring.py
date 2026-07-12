"""Hotel x aspect scoring.

Turns the labeled sentence instances into a defensible per-hotel, per-aspect sentiment
score. Three ideas stacked together:

  1. Weighted evidence  — every sentence instance carries a weight of
        recency_weight * trust_weight
     (older reviews and unverified reviews count for less, but are never dropped).

  2. Traveler-type conditioning — we keep the weighted sums broken out by traveler_type,
     so at recommendation time a *family* profile can up-weight what other families said
     without re-scanning 136k rows. This precomputation is what makes scoring all 50
     profiles instant.

  3. Empirical-Bayes shrinkage — a hotel with 3 spa mentions shouldn't outrank one with
     200 on noise, so each score is pulled toward the global mean for that aspect with a
     pseudo-count of K_SHRINKAGE.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config
from .taxonomy import ASPECT_KEYS, GENERAL_LABEL

INSTANCES_PATH = os.path.join(config.PROCESSED_DIR, "aspect_instances.parquet")
SCORES_PATH = os.path.join(config.PROCESSED_DIR, "hotel_aspect_scores.parquet")
HOTELS_PATH = os.path.join(config.PROCESSED_DIR, "hotels.parquet")


def build_aspect_instances(reviews: pd.DataFrame, sentence_inst: pd.DataFrame,
                           labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (review, sentence, aspect) with sentiment + weights attached.

    A sentence tagged with two aspects becomes two rows (each aspect gets the sentiment).
    `general` filler rows are kept but flagged, so they can feed semantic search while
    being excluded from aspect scoring.
    """
    lab = labels.set_index("sentence_text")
    inst = sentence_inst.merge(
        lab[["aspects", "polarity"]], on="sentence_text", how="left")
    inst["aspects"] = inst["aspects"].fillna(GENERAL_LABEL)
    inst["polarity"] = inst["polarity"].fillna(0.0)

    inst["aspect_list"] = inst["aspects"].str.split("|")
    exploded = inst.explode("aspect_list").rename(columns={"aspect_list": "aspect"})

    meta = reviews[["review_id", "hotel_id", "hotel_name", "hotel_category", "rating",
                    "review_date", "year_month", "season", "verified", "traveler_type",
                    "recency_weight", "trust_weight"]]
    exploded = exploded.merge(meta, on="review_id", how="left")
    exploded["weight"] = exploded["recency_weight"] * exploded["trust_weight"]
    exploded["is_general"] = exploded["aspect"] == GENERAL_LABEL
    return exploded.reset_index(drop=True)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = weights.sum()
    return float((values * weights).sum() / wsum) if wsum > 0 else np.nan


def build_hotel_aspect_scores(aspect_instances: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (hotel, aspect) with shrinkage, plus per-traveler-type weighted sums
    stored as columns so profile-time boosting needs no re-scan."""
    df = aspect_instances[~aspect_instances["is_general"]].copy()

    # global per-aspect prior (weighted mean sentiment across all hotels)
    priors = {a: _weighted_mean(g["polarity"].to_numpy(), g["weight"].to_numpy())
              for a, g in df.groupby("aspect")}

    traveler_types = ["business", "leisure", "family", "couple", "group", "solo", "unknown"]
    rows = []
    for (hotel_id, aspect), g in df.groupby(["hotel_id", "aspect"]):
        w = g["weight"].to_numpy()
        s = g["polarity"].to_numpy()
        W = float(w.sum())
        raw = _weighted_mean(s, w)
        prior = priors[aspect]
        shrunk = (W * raw + config.K_SHRINKAGE * prior) / (W + config.K_SHRINKAGE)

        pos_w = float(w[s >= config.POLARITY_THRESHOLD].sum())
        neg_w = float(w[s <= -config.POLARITY_THRESHOLD].sum())

        rec = {
            "hotel_id": hotel_id, "aspect": aspect,
            "n": int(len(g)), "evidence_weight": round(W, 4),
            "raw_score": round(raw, 4), "score": round(shrunk, 4),
            "pos_weight": round(pos_w, 4), "neg_weight": round(neg_w, 4),
        }
        # per traveler-type weighted sums (for profile-time conditioning)
        for tt in traveler_types:
            sub = g[g["traveler_type"] == tt]
            ww = sub["weight"].to_numpy()
            rec[f"wsum_{tt}"] = round(float(ww.sum()), 4)
            rec[f"wpol_{tt}"] = round(float((sub["polarity"].to_numpy() * ww).sum()), 4)
        rows.append(rec)

    scores = pd.DataFrame(rows)
    scores.attrs["priors"] = priors
    return scores


def build_hotel_table(reviews: pd.DataFrame, aspect_instances: pd.DataFrame) -> pd.DataFrame:
    """Per-hotel summary: identity + Bayesian-shrunk overall rating (normalized to [-1,1])."""
    global_rating = reviews["rating"].mean()
    rows = []
    for hotel_id, g in reviews.groupby("hotel_id"):
        n = len(g)
        shrunk_rating = (g["rating"].sum() + config.K_SHRINKAGE * global_rating) / (
            n + config.K_SHRINKAGE)
        rows.append({
            "hotel_id": hotel_id,
            "hotel_name": g["hotel_name"].iloc[0],
            "hotel_category": g["hotel_category"].iloc[0],
            "city": g["hotel_name"].iloc[0].split(",")[-1].strip(),
            "n_reviews": n,
            "mean_rating": round(g["rating"].mean(), 3),
            "shrunk_rating": round(shrunk_rating, 3),
            "rating_norm": round((shrunk_rating - 3.0) / 2.0, 4),  # 1..5 -> -1..1
            "verified_share": round(g["verified"].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("hotel_id").reset_index(drop=True)


def profile_conditioned_scores(scores: pd.DataFrame, traveler_type: str,
                               boost: float = config.TRAVELER_TYPE_MATCH_BOOST) -> pd.DataFrame:
    """Re-derive (hotel, aspect) scores with the profile's traveler_type up-weighted.
    Uses the precomputed per-type weighted sums — no row-level re-scan."""
    priors = scores.attrs.get("priors", {})
    types = ["business", "leisure", "family", "couple", "group", "solo", "unknown"]
    out = scores[["hotel_id", "aspect", "n"]].copy()

    W = np.zeros(len(scores))
    P = np.zeros(len(scores))
    for tt in types:
        mult = boost if tt == traveler_type else 1.0
        W = W + mult * scores[f"wsum_{tt}"].to_numpy()
        P = P + mult * scores[f"wpol_{tt}"].to_numpy()
    raw = np.divide(P, W, out=np.full_like(P, np.nan), where=W > 0)

    prior_vec = scores["aspect"].map(priors).to_numpy(dtype=float)
    shrunk = (W * np.nan_to_num(raw) + config.K_SHRINKAGE * prior_vec) / (
        W + config.K_SHRINKAGE)
    out["evidence_weight"] = np.round(W, 4)
    out["score"] = np.round(shrunk, 4)
    return out
