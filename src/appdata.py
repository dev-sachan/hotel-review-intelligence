"""Thin data-access layer for the Streamlit app.

The app never runs model inference — it only reads the cached artifacts the pipeline wrote.
Everything here is a cached loader so the UI is instant.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config, scoring, temporal
from .retrieval import EMB_PATH

PROFILES_OUT = os.path.join(config.PROCESSED_DIR, "profiles_enriched.json")
PRIORS_PATH = os.path.join(config.PROCESSED_DIR, "aspect_priors.json")


def artifacts_exist() -> bool:
    needed = [scoring.SCORES_PATH, scoring.HOTELS_PATH, scoring.INSTANCES_PATH,
              temporal.MONTHLY_PATH, temporal.DRIFT_PATH, temporal.CONTRA_PATH,
              PROFILES_OUT]
    return all(os.path.exists(p) for p in needed)


def load_all() -> dict:
    scores = pd.read_parquet(scoring.SCORES_PATH)
    priors = json.load(open(PRIORS_PATH, encoding="utf-8")) if os.path.exists(PRIORS_PATH) else {}
    scores.attrs["priors"] = priors
    d = {
        "scores": scores,
        "hotels": pd.read_parquet(scoring.HOTELS_PATH),
        "instances": pd.read_parquet(scoring.INSTANCES_PATH),
        "monthly": pd.read_parquet(temporal.MONTHLY_PATH),
        "drift": pd.read_parquet(temporal.DRIFT_PATH),
        "contradictions": pd.read_parquet(temporal.CONTRA_PATH),
        "profiles": json.load(open(PROFILES_OUT, encoding="utf-8")),
    }
    d["profiles_by_id"] = {p["profile_id"]: p for p in d["profiles"]}
    return d


def load_rich_output(profile_id: str) -> dict | None:
    path = os.path.join(config.OUTPUTS_DIR, "rich", f"{profile_id}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))
