"""Ingestion & validation: raw hackathon JSON -> clean typed DataFrames + the list of
unique sentences that the (expensive) NLP step actually needs to touch.

The key move here is deduplication. There are 136,451 sentence *instances* across the
50,000 reviews, but only 87 *unique* sentences. We split every review, collect the unique
set, and hand only those 87 to the classifier — the labels are joined back to every
instance later.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Normalize the mojibake em-dash that appears in the raw file (""/"�") to a real one,
# plus common encoding artifacts, so identical sentences dedupe correctly.
_FIXUPS = {"\x97": "—", "": "—", "�": "—"}


def _clean_text(text: str) -> str:
    for bad, good in _FIXUPS.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def load_reviews(path: str = config.RAW_REVIEWS_PATH) -> pd.DataFrame:
    """Load, validate and enrich the raw reviews. Fails loudly on schema violations."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    required = {"review_id", "hotel_id", "hotel_name", "hotel_category",
                "rating", "review_date", "review_text", "verified"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reviews missing required columns: {missing}")

    df["review_text"] = df["review_text"].astype(str).map(_clean_text)
    df = df[df["review_text"].str.len() > 0].copy()

    # rating must be an int in 1..5
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()
    df["rating"] = df["rating"].astype(int)

    # dates -> datetime + calendar features
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df[df["review_date"].notna()].copy()
    df["year"] = df["review_date"].dt.year
    df["month"] = df["review_date"].dt.month
    df["year_month"] = df["review_date"].dt.to_period("M").astype(str)
    df["season"] = df["month"].map(config.SEASON_OF_MONTH)

    df["verified"] = df["verified"].astype(bool)
    # traveler_type is present on ~40% of rows; keep the gaps as an explicit "unknown"
    if "traveler_type" not in df.columns:
        df["traveler_type"] = np.nan
    df["traveler_type"] = df["traveler_type"].fillna("unknown").astype(str)

    df = df.drop_duplicates(subset="review_id").reset_index(drop=True)

    # recency weight: exponential decay anchored to the newest review in the dataset
    anchor = df["review_date"].max()
    age_days = (anchor - df["review_date"]).dt.days.clip(lower=0)
    df["recency_weight"] = np.power(0.5, age_days / config.RECENCY_HALF_LIFE_DAYS)
    df["trust_weight"] = np.where(df["verified"], config.VERIFIED_WEIGHT,
                                  config.UNVERIFIED_WEIGHT)
    return df


def load_profiles(path: str = config.RAW_PROFILES_PATH) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["description"] = df["description"].astype(str).map(_clean_text)
    return df.reset_index(drop=True)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def build_sentence_instances(reviews: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Explode reviews into one row per (review, sentence). Returns the instance table and
    the sorted list of unique sentences the classifier must score."""
    rows = []
    for rid, text in zip(reviews["review_id"], reviews["review_text"]):
        for pos, sent in enumerate(split_sentences(text)):
            rows.append((rid, pos, sent))
    inst = pd.DataFrame(rows, columns=["review_id", "sentence_pos", "sentence_text"])
    unique_sentences = sorted(inst["sentence_text"].unique())
    return inst, unique_sentences
