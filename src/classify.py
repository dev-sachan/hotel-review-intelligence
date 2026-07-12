"""Aspect + sentiment classification over the 87 unique template sentences.

This is the ONLY heavy model inference in the whole pipeline, and it runs over 87 strings,
so it finishes in seconds on CPU. Two models:

  * zero-shot NLI (DeBERTa-v3-zeroshot) in multi-label mode -> which of the 15 aspects
    each sentence is about (a sentence can hit several, e.g. "kids loved the pool and the
    staff were wonderful" -> family_friendly + service).
  * a sentiment model (twitter-roberta) -> a signed polarity in [-1, 1].

The auto labels are written to `sentence_labels_auto.csv`. A hand-audited overrides file
(`sentence_overrides.csv`) is then layered on top to produce the authoritative
`sentence_labels.csv`. Because there are only 87 rows, the final mapping is 100% auditable.
"""
from __future__ import annotations

import os

import pandas as pd

from . import config
from .taxonomy import ASPECT_KEYS, GENERAL_LABEL, HYPOTHESES

AUTO_LABELS_PATH = os.path.join(config.PROCESSED_DIR, "sentence_labels_auto.csv")
OVERRIDES_PATH = os.path.join(config.PROJECT_ROOT, "data", "sentence_overrides.csv")
FINAL_LABELS_PATH = os.path.join(config.PROCESSED_DIR, "sentence_labels.csv")

_HYP_TO_ASPECT = dict(zip(HYPOTHESES, ASPECT_KEYS))


def classify_sentences(unique_sentences: list[str]) -> pd.DataFrame:
    """Run both models over the unique sentences. Heavy imports are local so that the
    Streamlit app (which never calls this) doesn't pay for torch/transformers at startup."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    zshot = pipeline("zero-shot-classification", model=config.ZERO_SHOT_MODEL, device=device)
    sentiment = pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL, device=device,
                         top_k=None, truncation=True)

    rows = []
    for sent in unique_sentences:
        z = zshot(sent, candidate_labels=HYPOTHESES, multi_label=True)
        scored = {_HYP_TO_ASPECT[lbl]: sc for lbl, sc in zip(z["labels"], z["scores"])}
        aspects = [a for a in ASPECT_KEYS if scored.get(a, 0.0) >= config.ASPECT_PRESENCE_THRESHOLD]
        if not aspects:
            aspects = [GENERAL_LABEL]

        s = {d["label"].lower(): d["score"] for d in sentiment(sent)[0]}
        # labels are positive / neutral / negative
        polarity = float(s.get("positive", 0.0) - s.get("negative", 0.0))

        rows.append({
            "sentence_text": sent,
            "aspects": "|".join(aspects),
            "polarity": round(polarity, 4),
            "top_aspect_score": round(max(scored.values()), 4),
            "aspect_scores": ";".join(f"{a}:{scored.get(a, 0):.2f}" for a in ASPECT_KEYS),
        })
    return pd.DataFrame(rows)


def apply_overrides(auto: pd.DataFrame) -> pd.DataFrame:
    """Layer the hand-audited overrides on top of the auto labels. Overrides can correct
    `aspects` and/or `polarity`; every override row carries a `reason` for the audit trail."""
    final = auto.copy()
    if not os.path.exists(OVERRIDES_PATH):
        final["overridden"] = False
        return final

    ov = pd.read_csv(OVERRIDES_PATH).set_index("sentence_text")
    final["overridden"] = False
    for idx, row in final.iterrows():
        st = row["sentence_text"]
        if st in ov.index:
            o = ov.loc[st]
            if isinstance(o.get("aspects"), str) and o["aspects"].strip():
                final.at[idx, "aspects"] = o["aspects"].strip()
            if "polarity" in ov.columns and pd.notna(o.get("polarity")):
                final.at[idx, "polarity"] = float(o["polarity"])
            final.at[idx, "overridden"] = True
    return final


def build_labels(unique_sentences: list[str], force: bool = False) -> pd.DataFrame:
    """Full label build with caching. Returns the authoritative sentence->labels table."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    if os.path.exists(AUTO_LABELS_PATH) and not force:
        auto = pd.read_csv(AUTO_LABELS_PATH)
        # if new sentences appeared, re-run
        if set(unique_sentences) - set(auto["sentence_text"]):
            auto = classify_sentences(unique_sentences)
            auto.to_csv(AUTO_LABELS_PATH, index=False)
    else:
        auto = classify_sentences(unique_sentences)
        auto.to_csv(AUTO_LABELS_PATH, index=False)

    final = apply_overrides(auto)
    final.to_csv(FINAL_LABELS_PATH, index=False)
    return final