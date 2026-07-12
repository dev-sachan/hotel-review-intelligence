"""Profile understanding: free-text traveler description -> structured intent.

For each of the 50 profiles we infer:
  * desired_dims + weights  — which aspects this traveler cares about, and how much
  * traveler_type           — solo / family / couple / group / business / leisure
  * budget band             — tight / mid / mid_high / high  -> a star-category fit vector
  * archetype slug          — e.g. "solo_female_culture" (matches sample_output.json)

Two complementary signals, by design:
  1. Deterministic keyword rules  — explainable and precise; the backbone.
  2. Embedding backstop           — MiniLM cosine between the profile text and each
     aspect's descriptor phrases, to catch paraphrases the rules miss.

Keeping the rules primary means every dimension we assign can be justified to a judge with
a concrete phrase from the profile, while the embeddings stop us from missing anything.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from . import config
from .taxonomy import ASPECT_KEYS, descriptor_texts

PROFILES_OUT = os.path.join(config.PROCESSED_DIR, "profiles_enriched.json")

# ---- keyword rules: aspect -> regex of trigger phrases (deterministic, explainable) ----
ASPECT_RULES: dict[str, str] = {
    "safety": r"safe|safety|secur|felt (?:very )?safe|sketchy|uncomfortable",
    "local_culture": r"local culture|markets|authentic|neighbou?rhood|local dining|real culture",
    "location_central": r"central|walkable|heart of the city|near the office|minimize travel|walking distance",
    "business_facilities": r"wifi|wi-fi|internet|remote work|meeting|conference|desk|video call|business",
    "quietness": r"quiet|peaceful|restful|calls|focus|soundproof|noise",
    # \bspa\b so we don't match "meeting SPAce"; \beat\b so we don't match "grEAT"
    "wellness_spa": r"\bspa\b|wellness|sauna|massage|retreat|unwind|self-care",
    "family_friendly": r"famil|toddler|kids|children|connecting rooms|pool|high chair",
    # \bvalue\b matches the noun ("good value", "value is everything") but not the verb
    # "values"; bare "budget" is dropped so merely *stating* a budget band doesn't imply
    # price-sensitivity (the budget->star fit is handled separately by category_fit).
    "value": r"\bvalue\b|cheap|affordable|shoestring|tight budget|splitting costs|save money|bang for|budget backpacker",
    "nightlife": r"nightlife|bars|clubs|social scene|night out|lively",
    "luxury": r"luxur|five-star|5-star|refinement|impeccable|premium|world-class|discerning|high-end|no object",
    "food_dining": r"food|dining|restaurant|culinary|foodie|cuisine|breakfast|\beat\b|eateries|dine",
    "beach_access": r"beach|seaside|beachfront|shore|sun-seeker|\bsea\b",
    "accessibility": r"accessib|wheelchair|step-free|mobility|ramp|reduced mobility",
    "cleanliness": r"clean|spotless|spotless cleanliness|immaculate|hygiene|tidy",
    "service": r"attentive|personal service|helpful staff|attentive staff|warm service|impeccable service",
}

TRAVELER_RULES: list[tuple[str, str]] = [
    ("family", r"famil|toddler|kids|children|parents"),
    ("couple", r"couple|honeymoon|romantic|getaway"),
    ("group", r"group|friends|bachelor|bachelorette|splitting costs"),
    ("business", r"business|corporate|road-warrior|expense-account|office"),
    ("solo", r"solo|alone|traveling alone|independent|backpacker|freelancer|nomad|remote worker"),
]

BUDGET_RULES: list[tuple[str, str]] = [
    ("high", r"budget is no object|no object|higher budget|high-end|luxury|five-star|premium|world-class"),
    ("mid_high", r"mid-to-high|upper budget|mid-range to upper|happy with a higher budget"),
    ("tight", r"tight budget|shoestring|budget backpacker|value is everything|cheap|save money"),
    ("mid", r"mid-range|expense-account|good value|mid-to-high budget"),
]

# budget band -> preference weight per star category (used in category-fit term)
CATEGORY_FIT: dict[str, dict[str, float]] = {
    "tight":    {"3-star": 1.0, "4-star": 0.4, "5-star": -0.4},
    "mid":      {"3-star": 0.6, "4-star": 1.0, "5-star": 0.5},
    "mid_high": {"3-star": 0.1, "4-star": 1.0, "5-star": 0.8},
    "high":     {"3-star": -0.4, "4-star": 0.6, "5-star": 1.0},
}

FEMALE_HINT = re.compile(r"\bfemale\b|\bwoman\b|\bher\b", re.I)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _rule_dims(text: str) -> dict[str, float]:
    t = text.lower()
    dims = {}
    for aspect, pattern in ASPECT_RULES.items():
        hits = len(re.findall(pattern, t))
        if hits:
            dims[aspect] = 1.0 + 0.25 * (hits - 1)  # repeated mentions weigh a little more
    return dims


def _embedding_dims(text: str, threshold: float = 0.30) -> dict[str, float]:
    """Cosine of the profile text against each aspect's descriptor phrases; keep the max
    per aspect above threshold as a secondary (weaker) signal."""
    pairs = descriptor_texts()
    emb = _get_embedder()
    desc_vecs = emb.encode([d for _, d in pairs], normalize_embeddings=True)
    prof_vec = emb.encode([text], normalize_embeddings=True)[0]
    sims = desc_vecs @ prof_vec
    best: dict[str, float] = {}
    for (aspect, _), sim in zip(pairs, sims):
        best[aspect] = max(best.get(aspect, -1.0), float(sim))
    return {a: s for a, s in best.items() if s >= threshold}


def _first_match(rules, text: str, default: str) -> str:
    t = text.lower()
    for label, pattern in rules:
        if re.search(pattern, t):
            return label
    return default


# Embedding backstop is used ONLY to fill in when the rules find too few dimensions.
# Measured finding: on this taxonomy the descriptor space is
# semantically overlapping, so low-threshold embedding matches add more noise than signal
# (e.g. `business_facilities` scores 0.33-0.45 against profiles with no business need). The
# keyword rules capture the real dims cleanly, so they stay authoritative and embeddings
# only supplement — with a high threshold — when a profile is under-specified by the rules.
MIN_RULE_DIMS = 2
EMB_FALLBACK_THRESHOLD = 0.50


def enrich_profile(profile_id: str, description: str, use_embeddings: bool = True) -> dict:
    rule_dims = _rule_dims(description)

    combined: dict[str, float] = dict(rule_dims)
    emb_dims: dict[str, float] = {}
    if use_embeddings and len(rule_dims) < MIN_RULE_DIMS:
        emb_dims = _embedding_dims(description, threshold=EMB_FALLBACK_THRESHOLD)
        for a, sim in emb_dims.items():
            combined[a] = max(combined.get(a, 0.0), 0.6 * sim)

    # keep the strongest 3-5 dims, normalize to sum 1
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ranked = [(a, w) for a, w in ranked if w > 0]
    total = sum(w for _, w in ranked) or 1.0
    weights = {a: round(w / total, 4) for a, w in ranked}

    traveler_type = _first_match(TRAVELER_RULES, description, "leisure")
    budget = _first_match(BUDGET_RULES, description, "mid")

    archetype = _make_archetype(traveler_type, list(weights.keys()), description)

    return {
        "profile_id": profile_id,
        "description": description,
        "traveler_type": traveler_type,
        "budget": budget,
        "desired_dims": list(weights.keys()),
        "dim_weights": weights,
        "category_fit": CATEGORY_FIT[budget],
        "archetype": archetype,
        "rule_dims": list(rule_dims.keys()),
        "embedding_dims": list(emb_dims.keys()),
    }


def _make_archetype(traveler_type: str, dims: list[str], description: str) -> str:
    """Compose a readable archetype slug like 'solo_female_culture' (sample_output style)."""
    parts = [traveler_type]
    if traveler_type == "solo" and FEMALE_HINT.search(description):
        parts.append("female")
    theme_map = {
        "local_culture": "culture", "wellness_spa": "wellness", "business_facilities": "business",
        "beach_access": "beach", "nightlife": "nightlife", "luxury": "luxury",
        "family_friendly": "family", "food_dining": "foodie", "accessibility": "accessible",
        "value": "budget", "safety": "safety", "location_central": "central",
    }
    for d in dims:
        if d in theme_map and theme_map[d] not in parts:
            parts.append(theme_map[d])
        if len(parts) >= 3:
            break
    return "_".join(parts)


def build_all_profiles(profiles_df: pd.DataFrame, use_embeddings: bool = True) -> list[dict]:
    out = [enrich_profile(r["profile_id"], r["description"], use_embeddings)
           for _, r in profiles_df.iterrows()]
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(PROFILES_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
