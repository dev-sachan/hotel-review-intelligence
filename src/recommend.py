"""The recommender: profile intent + hotel scores -> ranked, evidence-backed top-5.

The ranking is fully deterministic. For a profile with dimension weights w_d:

    fit(hotel) =  Σ_d  w_d * score(hotel, d)              # personalized aspect fit
               +  RATING_PRIOR_WEIGHT * rating_norm        # overall quality tie-breaker
               +  CATEGORY_FIT_WEIGHT * category_fit        # budget vs star band
               -  Σ_d  drift_penalty(hotel, d)             # confident worsening trends

where score(hotel, d) uses the profile's traveler_type-conditioned scores. A hotel must
have real evidence on the profile's top dimension (evidence_weight >= MIN_EVIDENCE_WEIGHT)
to be eligible at all — we never recommend on silence.

The LLM (if enabled) only ever *rephrases* the justification built from these numbers; it
cannot change the ranking or invent facts.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config
from .schema import (Evidence, RichOutput, RichRecommendation, SubmissionOutput, TopHotel)
from .scoring import profile_conditioned_scores
from .taxonomy import DISPLAY_NAMES


def _drift_lookup(drift: pd.DataFrame) -> dict:
    d = {}
    for _, r in drift.iterrows():
        d[(r["hotel_id"], r["aspect"])] = r
    return d


def rank_hotels(profile: dict, scores: pd.DataFrame, hotels: pd.DataFrame,
                drift: pd.DataFrame, top_n: int = config.TOP_N) -> pd.DataFrame:
    """Deterministic composite ranking for one profile."""
    weights = profile["dim_weights"]
    active = list(weights.keys())
    top_dim = active[0] if active else None

    cond = profile_conditioned_scores(scores, profile["traveler_type"])
    wide = cond.pivot(index="hotel_id", columns="aspect", values="score")
    ev = cond.pivot(index="hotel_id", columns="aspect", values="evidence_weight")

    hz = hotels.set_index("hotel_id")
    drift_map = _drift_lookup(drift)

    rows = []
    for hotel_id in hz.index:
        # eligibility: real evidence on the top desired dimension
        if top_dim is not None:
            if pd.isna(ev.get(top_dim, pd.Series()).get(hotel_id, np.nan)) or \
               ev.loc[hotel_id, top_dim] < config.MIN_EVIDENCE_WEIGHT:
                continue

        fit = 0.0
        aspect_contrib = {}
        for d, w in weights.items():
            sc = wide.loc[hotel_id, d] if d in wide.columns else np.nan
            if pd.isna(sc):
                sc = 0.0
            aspect_contrib[d] = round(float(sc), 4)
            fit += w * float(sc)

        cat_fit = profile["category_fit"].get(hz.loc[hotel_id, "hotel_category"], 0.0)
        fit += config.RATING_PRIOR_WEIGHT * float(hz.loc[hotel_id, "rating_norm"])
        fit += config.CATEGORY_FIT_WEIGHT * cat_fit

        # drift penalty: matched dims with a confident *declining* trend
        penalty = 0.0
        for d in active:
            row = drift_map.get((hotel_id, d))
            if row is not None and row["trend_dir"] == "declining" \
                    and row["trend_tier"] in config.DRIFT_PENALTY:
                penalty += config.DRIFT_PENALTY[row["trend_tier"]]
        fit -= penalty

        rows.append({
            "hotel_id": hotel_id,
            "hotel_name": hz.loc[hotel_id, "hotel_name"],
            "hotel_category": hz.loc[hotel_id, "hotel_category"],
            "composite": round(fit, 4),
            "display_score": config.display_score(fit),
            "aspect_contrib": aspect_contrib,
            "drift_penalty": round(penalty, 4),
        })

    ranked = pd.DataFrame(rows).sort_values("composite", ascending=False).head(top_n)
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def gather_evidence(hotel_id: str, active_aspects: list[str], aspect_instances: pd.DataFrame,
                    contradictions: pd.DataFrame, k: int = config.EVIDENCE_QUOTES_PER_HOTEL
                    ) -> tuple[list[dict], str | None]:
    """Pick the best supporting quotes for a hotel across the profile's aspects, and a
    contradiction-aware caveat. Prefers positive, recent, verified, on-aspect quotes; if an
    aspect is contested at this hotel, include one honest counter-quote."""
    pool = aspect_instances[(aspect_instances["hotel_id"] == hotel_id) &
                            (aspect_instances["aspect"].isin(active_aspects)) &
                            (~aspect_instances["is_general"])].copy()
    if pool.empty:
        return [], None

    pool["quality"] = (pool["polarity"] * pool["recency_weight"] *
                       pool["trust_weight"])
    pool = pool.sort_values("quality", ascending=False)

    picks, seen = [], set()
    for _, r in pool.iterrows():
        if r["sentence_text"] in seen or r["polarity"] <= 0.15:
            continue
        seen.add(r["sentence_text"])
        picks.append(r)
        if len(picks) >= k:
            break

    evidence = [{
        "review_id": r["review_id"],
        "quote": r["sentence_text"],
        "aspect": r["aspect"],
        "date": str(pd.to_datetime(r["review_date"]).date()),
        "verified": bool(r["verified"]),
        "sentiment": round(float(r["polarity"]), 3),
    } for r in picks]

    # caveat from contradictions on the matched aspects
    caveat = None
    contra = contradictions[(contradictions["hotel_id"] == hotel_id) &
                            (contradictions["aspect"].isin(active_aspects))]
    if not contra.empty:
        c = contra.sort_values("disagreement", ascending=False).iloc[0]
        name = DISPLAY_NAMES.get(c["aspect"], c["aspect"])
        caveat = f"Reviewers are split on {name.lower()}."
        if isinstance(c["resolution"], str) and c["resolution"].strip():
            caveat += " " + c["resolution"]
    return evidence, caveat


def build_recommendation(profile: dict, scores, hotels, drift, aspect_instances,
                         contradictions, narrator=None) -> RichOutput:
    ranked = rank_hotels(profile, scores, hotels, drift)
    active = profile["desired_dims"]
    recs = []
    for _, row in ranked.iterrows():
        evidence, caveat = gather_evidence(row["hotel_id"], active, aspect_instances,
                                           contradictions)
        if not evidence:  # eligibility guaranteed evidence on top dim, but be safe
            continue
        matched = [d for d in active if row["aspect_contrib"].get(d, 0) > 0.05]
        justification = _template_justification(profile, row, matched, evidence)
        if narrator is not None:
            justification = narrator.polish(profile, row, matched, evidence, justification)

        recs.append(RichRecommendation(
            rank=int(row["rank"]),
            hotel_id=row["hotel_id"],
            hotel_name=row["hotel_name"],
            hotel_category=row["hotel_category"],
            score=row["display_score"],
            matched_aspects=matched,
            aspect_scores={d: row["aspect_contrib"].get(d, 0.0) for d in active},
            justification=justification,
            supporting_evidence=[Evidence(**e) for e in evidence],
            caveats=caveat,
        ))

    return RichOutput(
        profile_id=profile["profile_id"],
        archetype=profile["archetype"],
        profile_summary=_profile_summary(profile),
        desired_dims=active,
        recommendations=recs,
    )


def _profile_summary(profile: dict) -> str:
    dims = ", ".join(DISPLAY_NAMES.get(d, d).lower() for d in profile["desired_dims"][:3])
    return (f"{profile['traveler_type'].capitalize()} traveler "
            f"({profile['budget'].replace('_', '-')} budget) prioritizing {dims}.")


def _template_justification(profile: dict, row, matched: list[str], evidence: list[dict]) -> str:
    """Deterministic justification composed from the real numbers and matched aspects.
    This is always produced and is the guaranteed fallback if LLM polishing is off/failed."""
    name = row["hotel_name"]
    if matched:
        strengths = ", ".join(f"{DISPLAY_NAMES.get(d, d).lower()} "
                              f"({row['aspect_contrib'].get(d, 0):+.2f})" for d in matched[:3])
        core = (f"{name} ranks #{int(row['rank'])} for this traveler on "
                f"{strengths}, weighted by what matters most to them.")
    else:
        core = f"{name} ranks #{int(row['rank'])} as a strong all-round match."
    ev_bits = evidence[0]["quote"] if evidence else ""
    return f"{core} Guests note: \"{ev_bits}\""


def to_submission(rich: RichOutput) -> SubmissionOutput:
    """Project the rich output down to the exact sample_output.json schema."""
    return SubmissionOutput(
        profile_id=rich.profile_id,
        archetype=rich.archetype,
        desired_dims=rich.desired_dims,
        top_hotels=[TopHotel(rank=r.rank, hotel_id=r.hotel_id, hotel_name=r.hotel_name,
                             hotel_category=r.hotel_category, score=r.score)
                    for r in rich.recommendations],
    )


def write_outputs(rich_outputs: list[RichOutput]) -> None:
    sub_dir = os.path.join(config.OUTPUTS_DIR, "submission")
    rich_dir = os.path.join(config.OUTPUTS_DIR, "rich")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(rich_dir, exist_ok=True)

    all_sub = []
    for rich in rich_outputs:
        sub = to_submission(rich)
        all_sub.append(sub.model_dump())
        with open(os.path.join(sub_dir, f"{rich.profile_id}.json"), "w", encoding="utf-8") as f:
            json.dump(sub.model_dump(), f, indent=2)
        with open(os.path.join(rich_dir, f"{rich.profile_id}.json"), "w", encoding="utf-8") as f:
            json.dump(rich.model_dump(), f, indent=2)

    with open(os.path.join(config.OUTPUTS_DIR, "all_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(all_sub, f, indent=2)
    with open(os.path.join(config.OUTPUTS_DIR, "all_profiles_rich.json"), "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in rich_outputs], f, indent=2)
