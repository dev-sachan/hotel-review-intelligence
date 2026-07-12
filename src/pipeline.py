"""End-to-end pipeline: raw JSON -> cached artifacts -> validated recommendation outputs.

Run:
    python -m src.pipeline            # use cached model labels/embeddings if present
    python -m src.pipeline --force    # recompute everything (re-runs the classifiers)
    python -m src.pipeline --polish   # also LLM-polish justifications (slower)

Every heavy artifact is cached under data/processed/, so re-runs and the Streamlit app are
fast. Timings are printed per stage.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from . import (classify, config, ingest, profiles, recommend, retrieval, scoring,
               temporal)

REVIEWS_PATH = os.path.join(config.PROCESSED_DIR, "reviews.parquet")


class _Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t = time.time(); print(f"[..] {self.label}"); return self
    def __exit__(self, *a): print(f"[ok] {self.label}  ({time.time()-self.t:.1f}s)")


def run(force: bool = False, polish: bool = False) -> None:
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    with _Timer("Ingest reviews + profiles"):
        reviews = ingest.load_reviews()
        profiles_df = ingest.load_profiles()
        sentence_inst, unique_sentences = ingest.build_sentence_instances(reviews)
        reviews.to_parquet(REVIEWS_PATH, index=False)
        print(f"     {len(reviews)} reviews, {len(sentence_inst)} sentence instances, "
              f"{len(unique_sentences)} unique sentences")

    with _Timer("Classify 87 sentences (aspects + sentiment)"):
        labels = classify.build_labels(unique_sentences, force=force)
        n_over = int(labels["overridden"].sum()) if "overridden" in labels else 0
        print(f"     {len(labels)} labeled, {n_over} hand-audited overrides applied")

    with _Timer("Build aspect instances + hotel/aspect scores"):
        aspect_instances = scoring.build_aspect_instances(reviews, sentence_inst, labels)
        aspect_instances.to_parquet(scoring.INSTANCES_PATH, index=False)
        hotels = scoring.build_hotel_table(reviews, aspect_instances)
        hotels.to_parquet(scoring.HOTELS_PATH, index=False)
        scores = scoring.build_hotel_aspect_scores(aspect_instances)
        scores.to_parquet(scoring.SCORES_PATH, index=False)
        # persist the priors alongside (needed by profile-conditioned scoring at app time)
        pd.Series(scores.attrs["priors"]).to_json(
            os.path.join(config.PROCESSED_DIR, "aspect_priors.json"))

    with _Timer("Temporal drift + contradictions"):
        monthly = temporal.build_monthly(reviews, aspect_instances)
        monthly.to_parquet(temporal.MONTHLY_PATH, index=False)
        drift = temporal.build_drift(aspect_instances, reviews)
        drift.to_parquet(temporal.DRIFT_PATH, index=False)
        contradictions = temporal.build_contradictions(aspect_instances)
        contradictions.to_parquet(temporal.CONTRA_PATH, index=False)
        print(f"     {len(drift)} drift tests, {len(contradictions)} contradiction flags")

    with _Timer("Sentence embeddings (semantic search)"):
        if force or not os.path.exists(retrieval.EMB_PATH):
            retrieval.build_sentence_embeddings(labels)

    with _Timer("Enrich 50 profiles"):
        enriched = profiles.build_all_profiles(profiles_df, use_embeddings=True)

    narrator = None
    if polish:
        with _Timer("Load local LLM for prose polish"):
            from .narrate import Narrator
            narrator = Narrator()

    with _Timer("Generate + validate recommendations (50 profiles)"):
        # scores loses its .attrs on parquet round-trip within this process we keep the
        # in-memory object, which still carries priors.
        rich_outputs = [recommend.build_recommendation(
            p, scores, hotels, drift, aspect_instances, contradictions, narrator)
            for p in enriched]
        recommend.write_outputs(rich_outputs)
        print(f"     wrote {len(rich_outputs)} profiles to {config.OUTPUTS_DIR}")

    print("\nDone. Launch the demo with:  streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute model labels/embeddings")
    ap.add_argument("--polish", action="store_true", help="LLM-polish justifications")
    args = ap.parse_args()
    run(force=args.force, polish=args.polish)
